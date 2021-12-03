"""accelpy OpenAcc C++ Accelerator module"""


from accelpy.accel import AccelBase, get_compilers
from ctypes import c_int32, c_int64, c_float, c_double

t_main = """
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

//int ACCELPY_OPENACC_NGANGS;
//int ACCELPY_OPENACC_NWORKERS;
//int ACCELPY_OPENACC_LENVECTOR;

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

    const int64_t ngangs = (*team_x) * (*team_y) * (*team_z);
    const int64_t nworkers = (*thread_x) * (*thread_y) * (*thread_z);
    const int64_t lenvec = (*assign_x) * (*assign_y) * (*assign_z);

    res = accelpy_kernel((*run_id), (*device), (*channel), ngangs, nworkers, lenvec);

    return res;
}}

extern "C" int64_t accelpy_stop() {{

    int64_t res;

{datadeletes}

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
    const unsigned int strides[{ndim}] = {strides};

    {dtype} & operator() ({oparg}) {{
        return data[{offset}];
    }}

    {dtype} operator() ({oparg}) const {{
        return data[{offset}];
    }}
}};
"""

t_h2acopy = """
extern "C" int64_t {funcname}(void * data) {{
    int64_t res;

    {varname}.data = ({dtype} *) data;

    #pragma acc enter data copyin({varname})
    #pragma acc enter data copyin({varname}.data[0:{size}])

    res = 0;

    return res;
}}
"""

t_h2amalloc = """
extern "C" int64_t {funcname}(void * data) {{
    int64_t res;


    {varname}.data = ({dtype} *) data;

    #pragma acc enter data copyin({varname})
    #pragma acc enter data create({varname}.data[0:{size}])

    res = 0;

    return res;
}}
"""

t_a2hcopy = """
extern "C" int64_t {funcname}(void * data) {{
    int64_t res;

    #pragma acc exit data copyout({varname}.data[0:{size}])

    res = 0;

    return res;
}}
"""

t_testfunc = """
extern "C" int64_t accelpy_test_run() {{
    int64_t res;
    
    #pragma acc parallel num_gangs(1) num_workers({size}) vector_length(1)
    #pragma acc loop gang worker vector
    for (int id = 0; id < {varin}.shape[0]; id++) {{
        {varout}(id) = {varin}(id);
    }}

    res = 0;

    return res;
}}
"""

t_kernel = """
extern "C" int64_t accelpy_kernel( \\
        const int64_t run_id, const int64_t device, const int64_t channel, \\
        const int64_t ngangs, const int64_t nworkers, const int64_t lenvector){{

    int64_t res;

    #pragma acc parallel num_gangs(ngangs) num_workers(nworkers) vector_length({lenvec})

    {order}

    res = 0;

    return res;
}}
"""

class OpenaccCppAccel(AccelBase):

    name = "openacc_cpp"
    lang = "cpp"

    # dtype: ( C type name, ctype )
    # TODO: dynamically generates on loading
    dtypemap = {
        "int32": ["int32_t", c_int32],
        "int64": ["int64_t", c_int64],
        "float32": ["float", c_float],
        "float64": ["double", c_double]
    }

    def get_shapestr(self, arg):
        return ",".join([str(s) for s in arg["data"].shape])

    def get_stridestr(self, arg):
        return ",".join([str(int(s//arg["data"].itemsize)) for s
                in arg["data"].strides])

    def gen_varclasses(self, inputs, outputs):

        varclasses = []
        for arg in self._testdata+inputs+outputs:

            ndim = arg["data"].ndim
            dtype = self.get_dtype(arg)

            oparg = ", ".join(["int dim%d"%d for d in range(ndim)])
            offset = "+".join(["strides[%d]*dim%d"%(d,d) for d in range(ndim)])

            varclasses_fmt = {
                "clsname": "type_" + arg["curname"],
                "size": str(arg["data"].size),
                "ndim": str(ndim),
                "shape": "{%s}" % self.get_shapestr(arg),
                "strides": "{%s}" % self.get_stridestr(arg),
                "dtype": dtype,
                "offset":offset,
                "oparg":oparg
            }

            varclasses.append(t_varclass.format(**varclasses_fmt))

        return "\n\n".join(varclasses)

#    def gen_vardefs(self, inputs, outputs):
#
#        out = []
#
#        for arg in inputs+outputs:
#            out.append("type_{name} {name} = type_{name}();".format(
#                        name=arg["curname"]))
#
#        return "\n".join(out)

    def gen_datacopies(self, inputs, outputs):

        out = []

        for input in inputs:
            dtype = self.get_dtype(input)
            funcname = self.getname_h2acopy(input)

            out.append("type_{name} {name} = type_{name}();".format(
                        name=input["curname"]))

            out.append(t_h2acopy.format(funcname=funcname, size=str(input["data"].size),
                        varname=input["curname"], dtype=dtype))

        for output in outputs:
            dtype = self.get_dtype(output)
            funcname = self.getname_h2amalloc(output)

            out.append("type_{name} {name} = type_{name}();".format(
                        name=output["curname"]))

            out.append(t_h2amalloc.format(funcname=funcname, size=str(output["data"].size),
                        varname=output["curname"], dtype=dtype))

        for output in outputs:
            funcname = self.getname_a2hcopy(output)

            out.append(t_a2hcopy.format(funcname=funcname, size=str(output["data"].size),
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

        out.append(t_h2acopy.format(funcname=funcname_in, size=str(input["data"].size),
                    varname=input["curname"], dtype=dtype_in))
        out.append(t_h2amalloc.format(funcname=funcname_out, size=str(output["data"].size),
                    varname=output["curname"], dtype=dtype_out))
        out.append(t_a2hcopy.format(funcname=funcname_a2h, size=str(output["data"].size),
                    varname=output["curname"]))

        out.append(t_testfunc.format(varin=input["curname"],
                    varout=output["curname"], size=str(input["data"].size)))

        return "\n".join(out)

    def gen_kernel(self, compiler):
        
        order =  self._order.get_section(self.name)
        lenvec = "1" if compiler.vendor=="pgi" else "lenvector"

        return t_kernel.format(order="\n".join(order.body), lenvec=lenvec)

    def gen_code(self, inputs, outputs, compiler):

        self._macros

        datadeletes = []

        for data in inputs:
            datadeletes.append("#pragma acc exit data delete(%s.data)" % data["curname"])
            datadeletes.append("#pragma acc exit data delete(%s)" % data["curname"])

        main_fmt = {
            "varclasses":self.gen_varclasses(inputs, outputs),
            "testcode":self.gen_testcode(),
            "datacopies":self.gen_datacopies(inputs, outputs),
            "kernel":self.gen_kernel(compiler),
            "datadeletes": "\n".join(datadeletes)
        }

        code = t_main.format(**main_fmt)

        return code

#    def get_vartype(self, arg, prefix=""):
#
#        dtype = self.get_dtype(arg)
#        return "%s%s_dim%d" % (prefix, dtype, arg["data"].ndim)

    def getname_h2amalloc(self, arg):
        return "accelpy_h2amalloc_%s" % arg["curname"]

    def getname_h2acopy(self, arg):
        return "accelpy_h2acopy_%s" % arg["curname"]

    def getname_a2hcopy(self, arg):
        return "accelpy_a2hcopy_%s" % arg["curname"]

OpenaccCppAccel.avails[OpenaccCppAccel.name] = OpenaccCppAccel
