"""accelpy C++ Accelerator module"""


from accelpy.accel import AccelBase, get_compilers
from ctypes import c_int32, c_int64, c_float, c_double

t_main = """
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

{varclasses}

{testcode}

{datacopies}

{kernel}

extern "C" int64_t accelpy_start(  \\
        int64_t * run_id, int64_t * device, int64_t * channel, \\
        int64_t * thread_x, int64_t * thread_y, int64_t * thread_z, \\
        int64_t * team_x, int64_t * team_y, int64_t * team_z, \\
        int64_t * assign_x, int64_t * assign_y, int64_t * assign_z) {{

    int64_t res;
    int64_t ACCELPY_WORKER_ID;

    ACCELPY_WORKER_ID = 0;

    res = accelpy_kernel(ACCELPY_WORKER_ID);

    return res;
}}

extern "C" int64_t accelpy_stop() {{

    int64_t res;

    res = 0;

    return res;
}}
"""

t_varclass = """
class {vartype} {{
public:
    {dtype} * data;
    int64_t * _attrs; // size, ndim, itemsize, shape, strides

    {dtype}& operator() ({oparg}) {{
        int64_t * s = &(_attrs[3+_attrs[1]]);
        return data[{offset}];
    }}
    {dtype} operator() ({oparg}) const {{
        int64_t * s = &(_attrs[3+_attrs[1]]);
        return data[{offset}];
    }}

    int size() {{
        return _attrs[0];
    }}

    int ndim() {{
        return _attrs[1];
    }}

    int itemsize() {{
        return _attrs[2];
    }}

    int shape(int dim) {{
        return _attrs[3+dim];
    }}

    int stride(int dim) {{
        return _attrs[3+_attrs[1]+dim];
    }}

    int unravel_index(int tid, int dim) {{
        int64_t q, r=tid, s;
        for (int i = 0; i < dim + 1; i++) {{
            s = stride(i);
            q = r / s;
            r = r % s;
        }}

        return q;
    }}
}};
"""

t_h2a = """
extern "C" int64_t {funcname}(int64_t * attrsize, int64_t * _attrs, void * data) {{
    int64_t res;

    {varname}.data = ({dtype} *) data;
    {varname}._attrs = _attrs;
    //{varname}._attrs = (int64_t *) malloc((*attrsize) * sizeof(int64_t));
    //memcpy({varname}._attrs, _attrs, (*attrsize) * sizeof(int64_t));

    res = 0;

    return res;
}}
"""

t_a2hcopy = """
extern "C" int64_t {funcname}(int64_t * attrsize, int64_t * _attrs, void * data) {{
    int64_t res;

    res = 0;

    return res;
}}
"""

t_testfunc = """
extern "C" int64_t accelpy_test_run() {{
    int64_t res;
    
    for (int id = 0; id < {varin}.shape(0); id++) {{
        {varout}(id) = {varin}(id);
    }}

    //free({varin}._attrs);
    //free({varout}._attrs);

    res = 0;

    return res;
}}
"""

t_kernel = """
extern "C" int64_t accelpy_kernel(int64_t ACCELPY_WORKER_ID){{

    int64_t res;

    {order}

    res = 0;

    return res;
}}
"""

class CppAccel(AccelBase):

    name = "cpp"
    lang = "cpp"

    # dtype: ( C type name, ctype )
    # TODO: dynamically generates on loading
    dtypemap = {
        "int32": ["int32_t", c_int32],
        "int64": ["int64_t", c_int64],
        "float32": ["float", c_float],
        "float64": ["double", c_double]
    }

    def gen_varclasses(self, inputs, outputs):

        varclasses = []
        types = {}

        for arg in self._testdata+inputs+outputs:

            ndim = arg["data"].ndim
            dtype = self.get_dtype(arg)

            if dtype not in types:
                dtypes = {}
                types[dtype] = dtypes
            else:
                dtypes = types[dtype]

            if ndim not in dtypes:

                oparg = ", ".join(["int dim%d"%d for d in range(ndim)])
                offset = "+".join(["s[%d]*dim%d"%(d,d) for d in range(ndim)])

                varclasses_fmt = {
                    "vartype": self.get_vartype(arg),
                    "dtype": dtype,
                    "offset":offset,
                    "oparg":oparg
                }

                varclasses.append(t_varclass.format(**varclasses_fmt))

                dtypes[ndim] = True

        return "\n\n".join(varclasses)

    def gen_vardefs(self, inputs, outputs):

        out = []

        for arg in inputs+outputs:

            ndim, dname = self.get_argpair(arg)
            vartype = self.get_vartype(arg)

            out.append("%s %s = %s();" % (vartype, arg["curname"], vartype))

        return "\n".join(out)

    def gen_datacopies(self, inputs, outputs):

        out = []

        for input in inputs:
            vartype = self.get_vartype(input)
            dtype = self.get_dtype(input)
            funcname = self.getname_h2acopy(input)

            out.append("%s %s = %s();" % (vartype, input["curname"], vartype))
            out.append(t_h2a.format(funcname=funcname, varname=input["curname"], dtype=dtype))

        for output in outputs:
            vartype = self.get_vartype(output)
            dtype = self.get_dtype(output)
            funcname = self.getname_h2amalloc(output)

            out.append("%s %s = %s();" % (vartype, output["curname"], vartype))
            out.append(t_h2a.format(funcname=funcname, varname=output["curname"], dtype=dtype))

        for output in outputs:
            funcname = self.getname_a2hcopy(output)

            out.append(t_a2hcopy.format(funcname=funcname))

        return "\n".join(out)

    def gen_testcode(self):

        out = []

        input = self._testdata[0]
        output = self._testdata[1]

        vartype_in = self.get_vartype(input)
        vartype_out = self.get_vartype(output)
        dtype_in = self.get_dtype(input)
        dtype_out = self.get_dtype(input)
        funcname_in = "accelpy_test_h2acopy"
        funcname_out = "accelpy_test_h2amalloc"
        funcname_a2h = "accelpy_test_a2hcopy"

        out.append("%s %s = %s();" % (vartype_in, input["curname"], vartype_in))
        out.append("%s %s = %s();" % (vartype_out, output["curname"], vartype_out))

        out.append(t_h2a.format(funcname=funcname_in, varname=input["curname"], dtype=dtype_in))
        out.append(t_h2a.format(funcname=funcname_out, varname=output["curname"], dtype=dtype_out))
        out.append(t_a2hcopy.format(funcname=funcname_a2h))

        out.append(t_testfunc.format(varin=input["curname"], varout=output["curname"]))

        return "\n".join(out)

    def gen_kernel(self):

        order =  self._order.get_section(self.name)
        return t_kernel.format(order="\n".join(order.body))

    def gen_code(self, inputs, outputs):

        main_fmt = {
            "varclasses":self.gen_varclasses(inputs, outputs),
            "testcode":self.gen_testcode(),
            "datacopies":self.gen_datacopies(inputs, outputs),
            "kernel":self.gen_kernel()
        }

        code = t_main.format(**main_fmt)

        return code

    def get_vartype(self, arg, prefix=""):

        dtype = self.get_dtype(arg)
        return "%s%s_dim%d" % (prefix, dtype, arg["data"].ndim)

    def getname_h2amalloc(self, arg):
        return "accelpy_h2amalloc_%s" % arg["curname"]

    def getname_h2acopy(self, arg):
        return "accelpy_h2acopy_%s" % arg["curname"]

    def getname_a2hcopy(self, arg):
        return "accelpy_a2hcopy_%s" % arg["curname"]

AccelBase.avails[CppAccel.name] = CppAccel
