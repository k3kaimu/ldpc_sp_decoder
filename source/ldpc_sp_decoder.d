module ldpc_sp_decoder;

import std.algorithm : min, max;
import std.experimental.allocator.mallocator;

import core.simd;

version(LDC)
{
    import ldc.attributes;

    pragma(LDC_intrinsic, "llvm.minnum.v4f32")
    float4 _mm_minps_(float4, float4) pure @safe;
    pragma(LDC_intrinsic, "llvm.maxnum.v4f32")
    float4 _mm_maxps_(float4, float4) pure @safe;
    pragma(LDC_intrinsic, "llvm.minnum.v8f32")
    float8 _mm_minps_(float8, float8) pure @safe;
    pragma(LDC_intrinsic, "llvm.maxnum.v8f32")
    float8 _mm_maxps_(float8, float8) pure @safe;
}

version(DigitalMars)
{
    float4 _mm_minps_(float4 x, float4 y) pure @safe { return simd!(XMM.MINPS)(x, y); }
    float4 _mm_maxps_(float4 x, float4 y) pure @safe { return simd!(XMM.MAXPS)(x, y); }
    float8 _mm_minps_(float8 x, float8 y) pure @safe { return simd!(XMM.MINPS)(x, y); }
    float8 _mm_maxps_(float8 x, float8 y) pure @safe { return simd!(XMM.MAXPS)(x, y); }
    enum int fastmath = 0;
}


@fastmath
V vecminmax(V, F)(V v, F vmin_, F vmax_) @trusted
{
    static if(is(V : F))
    {
        import std.algorithm : min, max;
        return max(min(v, vmax_), vmin_);
    }
    else static if(is(typeof(_mm_minps_(v, v))))
    {
        V vmin = vmin_,
          vmax = vmax_;

        return _mm_maxps_(_mm_minps_(v, vmax), vmin);
    }
    else
    {
        import std.algorithm : min, max;
        foreach(i; 0 .. V.length) {
            v[i] = max(min(v[i], vmax_), vmin_);
        }

        return v;
    }
}


struct SpDecoderWorkspace(T)
{
    this(size_t N, in uint[][] row_mat)
    {
        maxRW = 0;
        size_t totElem = 0;
        foreach(row; row_mat) {
            maxRW = max(maxRW, row.length);
            totElem += row.length;
        }

        static if(is(T : float))
        {
            _allocate(edge_mat, totElem);
            _allocate(updated_p0p1, N);
            _allocate(input_p0p1_copy, N);
            _allocate(prob1m2p, maxRW);
            _allocate(decoded_cw, N);
        }
        else
        {
            _alignedAllocate(edge_mat, totElem);
            _alignedAllocate(updated_p0p1, N);
            _alignedAllocate(input_p0p1_copy, N);
            _alignedAllocate(prob1m2p, maxRW);
            _allocate(decoded_cw, N * T.length);
        }
    }


    @disable this(this);


    ~this()
    {
        _deallocate(edge_mat);
        _deallocate(updated_p0p1);
        _deallocate(input_p0p1_copy);
        _deallocate(prob1m2p);
        _deallocate(decoded_cw);
    }


    size_t maxRW;

    static if(is(T : float))
        alias E = T;
    else
        alias E = Vector!T;

    E[] edge_mat;
    E[] updated_p0p1;
    E[] input_p0p1_copy;
    E[] prob1m2p;
    ubyte[] decoded_cw;


  private:
    alias _allocator = AlignedMallocator.instance;


    void _alignedAllocate(ref E[] v, size_t n) @trusted
    {
        v = (cast(E*)_allocator.alignedAllocate(n * E.sizeof, E.sizeof))[0 .. n];
    }


    void _allocate(U)(ref U[] v, size_t n) @trusted
    {
        v = (cast(U*)_allocator.allocate(n * U.sizeof))[0 .. n];
    }


    void _deallocate(U)(ref U[] v) @trusted
    {
        _allocator.deallocate(v);
        v = null;
    }
}


@fastmath
void sumProductDecodeP0P1SIMD(V, F)(ref SpDecoderWorkspace!V ws, in uint[][] _row_mat, in F[] input_p0p1, uint max_iter)
{
    with(ws){
    import core.simd;

    static if(is(V : float)) {
        enum size_t P = 1;
        alias VecType = V;
    }else{
        enum size_t P = V.length;
        alias VecType = Vector!V;
    }

    immutable size_t _M = _row_mat.length;
    immutable size_t _N = input_p0p1.length / P;

    bool[P] success;

    static if(P != 1) {
        foreach(i; 0 .. P) foreach(j; 0 .. _N) {
            input_p0p1_copy[j][i] = input_p0p1[i*_N + j];
        }
    } else {
        foreach(i; 0 .. _N) {
            input_p0p1_copy[i] = input_p0p1[i];
        }
    }

    foreach(i; 0 .. _N) {
        input_p0p1_copy[i] = vecminmax(input_p0p1_copy[i], 0, 2.0^^25);
    }

    updated_p0p1[] = input_p0p1_copy[];

    edge_mat[] = 1;

    if(max_iter == 0) {
        // max_iter == 0のときはすぐに復号してそのままreturnする
        foreach(i; 0 .. P){
            static if(P == 1) {
                foreach(j; 0 .. _N)
                    decoded_cw[j] = updated_p0p1[j] > 1 ? 0 : 1;
            } else {
                foreach(j; 0 .. _N)
                    decoded_cw[i*_N + j] = updated_p0p1[j][i] > 1 ? 0 : 1;
            }
        }

        return;
    }


    foreach(iter; 0 .. max_iter) 
    {
        {
            auto p_last_edge_mat = edge_mat.ptr;
            auto p_edge_mat = edge_mat.ptr;
            foreach(i_row, row; _row_mat) {
                VecType prob_product = 1;
                foreach(i_col_index1, i_col_1; row) {
                    VecType q1 = updated_p0p1[i_col_1];
                    VecType q2 = *p_last_edge_mat;
                    ++p_last_edge_mat;

                    VecType p1m2p = (q1 - q2) / (q1 + q2);
                    prob1m2p[i_col_index1] = p1m2p;
                    prob_product *= p1m2p;
                }

                foreach(i_col_index1, i_col_1; row) {
                    VecType p1m2p = prob1m2p[i_col_index1];
                    VecType tmp = (p1m2p + prob_product) / (p1m2p - prob_product);
                    *p_edge_mat = vecminmax(tmp, 0.5^^15, 2.0^^15);
                    ++p_edge_mat;
                }
            }
        }

        updated_p0p1[] = input_p0p1_copy[];

        {
            auto p_edge_mat = edge_mat.ptr;
            foreach(i_row, row; _row_mat) {
                foreach(i_col_index, i_col; row) {
                    updated_p0p1[i_col] *= *p_edge_mat;
                    ++p_edge_mat;
                }
            }
        }

        foreach(i; 0 .. _N) {
            updated_p0p1[i] = vecminmax(updated_p0p1[i], 0, 2.0^^25);
        }

        foreach(i; 0 .. P){
            if(success[i]) continue;

            static if(P == 1) {
                foreach(j; 0 .. _N)
                    decoded_cw[j] = updated_p0p1[j] > 1 ? 0 : 1;
            } else {
                foreach(j; 0 .. _N)
                    decoded_cw[i*_N + j] = updated_p0p1[j][i] > 1 ? 0 : 1;
            }

            if(checkCodeword(_row_mat, decoded_cw[i*_N .. (i+1)*_N]))
                success[i] = true;
        }

        bool checkAllSuccess = true;
        foreach(i; 0 .. P)
            checkAllSuccess = checkAllSuccess && success[i];

        if(checkAllSuccess)
            break;
    }
    }
}


bool checkCodeword(T)(in uint[][] _row_mat, in T[] decoded_cw) @safe
if(is(T: ubyte))
{
    immutable _M = _row_mat.length;

    bool check = true;
    foreach(i_check; 0 .. _M) {
        ubyte c;
        foreach(i_col; _row_mat[i_check])
            c += decoded_cw[i_col];
        
        if(c % 2 == 1) {
            check = false;
            break;
        }
    }

    return check;
}
