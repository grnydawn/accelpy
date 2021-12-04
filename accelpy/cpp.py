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

extern "C" int64_t accelpy_start() {{

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
class {clsname} {{
public:
    {dtype} * data;

    const unsigned int size = {size};
    const unsigned int ndim = {ndim};
    const unsigned int shape[{ndim}] = {shape};
    const unsigned int stride[{ndim}] = {stride};

    {dtype} & operator() ({oparg}) {{
        return data[{offset}];
    }}

    {dtype} operator() ({oparg}) const {{
        return data[{offset}];
    }}

    int unravel_index(unsigned int tid, unsigned int dim) {{
        unsigned int q, r=tid, s;
        for (unsigned int i = 0; i < dim + 1; i++) {{
            s = stride[i];
            q = r / s;
            r = r % s;
        }}

        return q;
    }}
}};
"""

t_h2a = """
extern "C" int64_t {funcname}(void * data) {{
    int64_t res;

    {varname}.data = ({dtype} *) data;

    res = 0;

    return res;
}}
"""

t_a2h = """
extern "C" int64_t {funcname}(void * data) {{
    int64_t res;

    res = 0;

    return res;
}}
"""

t_testfunc = """
extern "C" int64_t accelpy_test_run() {{
    int64_t res;
    
    for (int id = 0; id < {varin}.shape[0]; id++) {{
        {varout}(id) = {varin}(id);
    }}

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
        for arg in self._testdata+inputs+outputs:

            ndim = arg["data"].ndim
            dtype = self.get_dtype(arg)

            oparg = ", ".join(["int dim%d"%d for d in range(ndim)])
            offset = "+".join(["stride[%d]*dim%d"%(d,d) for d in range(ndim)])

            varclasses_fmt = {
                "clsname": "type_" + arg["curname"],
                "size": str(arg["data"].size),
                "ndim": str(ndim),
                "shape": "{%s}" % self.get_shapestr(arg),
                "stride": "{%s}" % self.get_stridestr(arg),
                "dtype": dtype,
                "offset":offset,
                "oparg":oparg
            }

            varclasses.append(t_varclass.format(**varclasses_fmt))

        return "\n\n".join(varclasses)

    def gen_datacopies(self, inputs, outputs):

        out = []

        for input in inputs:
            dtype = self.get_dtype(input)
            funcname = self.getname_h2acopy(input)

            out.append("type_{name} {name} = type_{name}();".format(
                        name=input["curname"]))

            out.append(t_h2a.format(funcname=funcname,
                        varname=input["curname"], dtype=dtype))

        for output in outputs:
            dtype = self.get_dtype(output)
            funcname = self.getname_h2amalloc(output)

            out.append("type_{name} {name} = type_{name}();".format(
                        name=output["curname"]))

            out.append(t_h2a.format(funcname=funcname,
                        varname=output["curname"], dtype=dtype))

        for output in outputs:
            funcname = self.getname_a2hcopy(output)

            out.append(t_a2h.format(funcname=funcname,
                        varname=output["curname"]))

        return "\n".join(out)

    def gen_testcode(self):

        out = []

        input = self._testdata[0]
        output = self._testdata[1]

        dtype_in = self.get_dtype(input)
        dtype_out = self.get_dtype(input)
        funcname_in = "accelpy_test_h2acopy"
        funcname_out = "accelpy_test_h2amalloc"
        funcname_a2h = "accelpy_test_a2hcopy"

        out.append("type_{name} {name} = type_{name}();".format(
                    name=input["curname"]))

        out.append("type_{name} {name} = type_{name}();".format(
                    name=output["curname"]))

        out.append(t_h2a.format(funcname=funcname_in,
                    varname=input["curname"], dtype=dtype_in))
        out.append(t_h2a.format(funcname=funcname_out,
                    varname=output["curname"], dtype=dtype_out))
        out.append(t_a2h.format(funcname=funcname_a2h,
                    varname=output["curname"]))

        out.append(t_testfunc.format(varin=input["curname"],
                    varout=output["curname"]))

        return "\n".join(out)

    def gen_kernel(self):

        order =  self._order.get_section(self.name)
        return t_kernel.format(order="\n".join(order.body))

    def gen_code(self, compiler, inputs, outputs, worker_triple, run_id, device, channel):

        macros = {}

        main_fmt = {
            "varclasses":self.gen_varclasses(inputs, outputs),
            "testcode":self.gen_testcode(),
            "datacopies":self.gen_datacopies(inputs, outputs),
            "kernel":self.gen_kernel()
        }

        code = t_main.format(**main_fmt)

        return code, macros

    def getname_h2amalloc(self, arg):
        return "accelpy_h2amalloc_%s" % arg["curname"]

    def getname_h2acopy(self, arg):
        return "accelpy_h2acopy_%s" % arg["curname"]

    def getname_a2hcopy(self, arg):
        return "accelpy_a2hcopy_%s" % arg["curname"]

CppAccel.avails[CppAccel.name] = CppAccel
