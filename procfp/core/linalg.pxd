cdef:
    void ldl(double[:, ::1] x, double[:, ::1] w, double[::1] d)
    void solve_linear(double[:, ::1] ml, double[::1] d, double[::1] z, double[::1] x)