"""accelpy Hip Accelerator module"""

# TODO: prevent maxlimit violations such as max thread per block
# TODO: expose current(possibly modified) configuration from workers/teams/assignments
# TODO: load cudart shared lib and uses several apis
# TODO: support multi-device and streams


from accelpy.accel import AccelBase, get_compilers
from ctypes import c_int32, c_int64, c_float, c_double

t_main = """
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <hip/hip_runtime.h>

{varclasses}

{testcode}

{datacopies}

{kernel}

extern "C" int64_t accelpy_start() {{

    int64_t res;

    const dim3 teams = dim3(ACCELPY_HIP_TEAM_ARGS);
    const dim3 workers = dim3(ACCELPY_HIP_WORKER_ARGS);

    accelpy_kernel<<<teams, workers>>>({launch_args});

    res = 0;

    return res;
}}

extern "C" int64_t accelpy_stop() {{

    int64_t res;

    res = 0;

    return res;
}}
"""

t_varclass = """
class dev_{clsname} {{
public:
    {dtype} * data;

    const unsigned int size = {size};
    const unsigned int ndim = {ndim};
    const unsigned int shape[{ndim}] = {shape};
    const unsigned int stride[{ndim}] = {stride};

    __device__ {dtype} & operator() ({oparg}) {{
        return data[{offset}];
    }}

    __device__ {dtype} operator() ({oparg}) const {{
        return data[{offset}];
    }}

    __device__ int unravel_index(int tid, int dim) {{
        int q, r=tid, s;
        for (int i = 0; i < dim + 1; i++) {{
            s = stride[i];
            q = r / s;
            r = r % s;
        }}

        return q;
    }}
}};
"""

t_h2acopy = """
extern "C" int64_t {funcname}(void * data) {{
    int64_t res;

    hipMalloc((void **)&(dev_{varname}.data), {size} * sizeof({dtype}));
    hipMemcpyHtoD(dev_{varname}.data, data, {size} * sizeof({dtype}));

    res = 0;

    return res;
}}
"""

t_h2amalloc = """
extern "C" int64_t {funcname}(void * data) {{
    int64_t res;

    hipMalloc((void **)&(dev_{varname}.data), {size} * sizeof({dtype}));

    res = 0;

    return res;
}}
"""

t_a2hcopy = """
extern "C" int64_t {funcname}(void * data) {{
    int64_t res;

    hipMemcpyDtoH(data, dev_{varname}.data, {size} * sizeof({dtype}));

    res = 0;

    return res;
}}
"""

t_testfunc = """
__global__ void accelpy_test_kernel(dev_type_{varin} {varin}, dev_type_{varout} {varout}){{

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < {varin}.size) {varout}(id) = {varin}(id);

}}

extern "C" int64_t accelpy_test_run() {{
    int64_t res;

    accelpy_test_kernel<<<1, {nworkers}>>>(dev_{varin}, dev_{varout});

    res = 0;

    return res;
}}
"""

t_kernel = """
__global__ void accelpy_kernel({kernel_args}){{

    {order}
}}
"""

class HipAccel(AccelBase):

    name = "hip"
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
#
#    def gen_varclasses(self, inputs, outputs):
#
#        varclasses = []
#        types = {}
#
#        for arg in self._testdata+inputs+outputs:
#
#            ndim = arg["data"].ndim
#            dtype = self.get_dtype(arg)
#
#            if dtype not in types:
#                dtypes = {}
#                types[dtype] = dtypes
#            else:
#                dtypes = types[dtype]
#
#            if ndim not in dtypes:
#
#                oparg = ", ".join(["int dim%d"%d for d in range(ndim)])
#                offset = "+".join(["s[%d]*dim%d"%(d,d) for d in range(ndim)])
#
#                varclasses_fmt = {
#                    "vartype": self.get_vartype(arg),
#                    "dtype": dtype,
#                    "offset":offset,
#                    "oparg":oparg
#                }
#
#                varclasses.append(t_varclass.format(**varclasses_fmt))
#
#                dtypes[ndim] = True
#
#        return "\n\n".join(varclasses)

#    def gen_vardefs(self, inputs, outputs):
#
#        out = []
#
#        for arg in inputs+outputs:
#
#            ndim, dname = self.get_argpair(arg)
#            vartype = self.get_vartype(arg)
#
#            out.append("dev_%s dev_%s = dev_%s();" % (vartype, arg["curname"], vartype))
#
#        return "\n".join(out)

    def gen_datacopies(self, inputs, outputs):

        out = []

        for input in inputs:
            dtype = self.get_dtype(input)
            funcname = self.getname_h2acopy(input)

            out.append("dev_type_{name} dev_{name} = dev_type_{name}();".format(
                        name=input["curname"]))

            out.append(t_h2acopy.format(funcname=funcname, size=str(input["data"].size),
                        varname=input["curname"], dtype=dtype))

        for output in outputs:
            dtype = self.get_dtype(output)
            funcname = self.getname_h2amalloc(output)

            out.append("dev_type_{name} dev_{name} = dev_type_{name}();".format(
                        name=output["curname"]))

            out.append(t_h2amalloc.format(funcname=funcname, size=str(output["data"].size),
                        varname=output["curname"], dtype=dtype))

        for output in outputs:
            dtype = self.get_dtype(output)
            funcname = self.getname_a2hcopy(output)

            out.append(t_a2hcopy.format(funcname=funcname, size=str(output["data"].size),
                        varname=output["curname"], dtype=dtype))

        return "\n".join(out)

#
#    def gen_datacopies(self, inputs, outputs):
#
#        out = []
#
#        for input in inputs:
#            vartype = self.get_vartype(input)
#            dtype = self.get_dtype(input)
#            funcname = self.getname_h2acopy(input)
#
#            out.append("dev_%s dev_%s = dev_%s();" % (vartype, input["curname"], vartype))
#            out.append(t_h2acopy.format(funcname=funcname, varname=input["curname"], dtype=dtype))
#
#        for output in outputs:
#            vartype = self.get_vartype(output)
#            dtype = self.get_dtype(output)
#            funcname = self.getname_h2amalloc(output)
#
#            out.append("dev_%s dev_%s = dev_%s();" % (vartype, output["curname"], vartype))
#            out.append(t_h2amalloc.format(funcname=funcname, varname=output["curname"],
#                        dtype=dtype))
#
#        for output in outputs:
#            funcname = self.getname_a2hcopy(output)
#            dtype = self.get_dtype(output)
#
#            out.append(t_a2hcopy.format(funcname=funcname, varname=output["curname"],
#                        dtype=dtype))
#
#        return "\n".join(out)

    def gen_testcode(self):

        out = []

        input = self._testdata[0]
        output = self._testdata[1]

        dtype_in = self.get_dtype(input)
        dtype_out = self.get_dtype(output)
        funcname_in = "accelpy_test_h2acopy"
        funcname_out = "accelpy_test_h2amalloc"
        funcname_a2h = "accelpy_test_a2hcopy"

        out.append("dev_type_{name} dev_{name} = dev_type_{name}();".format(
                    name=input["curname"]))

        out.append("dev_type_{name} dev_{name} = dev_type_{name}();".format(
                    name=output["curname"]))

        out.append(t_h2acopy.format(funcname=funcname_in, size=str(input["data"].size),
                    varname=input["curname"], dtype=dtype_in))
        out.append(t_h2amalloc.format(funcname=funcname_out, size=str(output["data"].size),
                    varname=output["curname"], dtype=dtype_out))
        out.append(t_a2hcopy.format(funcname=funcname_a2h, size=str(output["data"].size),
                    varname=output["curname"], dtype=dtype_out))

        out.append(t_testfunc.format(varin=input["curname"],
                    varout=output["curname"], nworkers=str(input["data"].size)))

        return "\n".join(out)

    def gen_kernel(self, inputs, outputs):

        args = []
        order =  self._order.get_section(self.name)

        for data in inputs+outputs:
            args.append("dev_type_{name} {name}".format(name=data["curname"]))

        return t_kernel.format(order="\n".join(order.body), kernel_args=", ".join(args))

    def gen_code(self, compiler, inputs, outputs, triple, run_id, device, channel):

        macros = {
            "ACCELPY_HIP_RUNID": str(run_id),
            "ACCELPY_HIP_TEAM_ARGS": ",".join([str(t) for t in triple[0]]),
            "ACCELPY_HIP_WORKER_ARGS": ",".join([str(t) for t in triple[1]]),
        }


        args = []
        for data in inputs+outputs:
            args.append("dev_%s" % data["curname"])

        main_fmt = {
            "varclasses":self.gen_varclasses(inputs, outputs),
            "testcode":self.gen_testcode(),
            "datacopies":self.gen_datacopies(inputs, outputs),
            "kernel":self.gen_kernel(inputs, outputs),
            "launch_args": ", ".join(args)
        }

        code = t_main.format(**main_fmt)

        return code, macros

    def getname_h2amalloc(self, arg):
        return "accelpy_h2amalloc_%s" % arg["curname"]

    def getname_h2acopy(self, arg):
        return "accelpy_h2acopy_%s" % arg["curname"]

    def getname_a2hcopy(self, arg):
        return "accelpy_a2hcopy_%s" % arg["curname"]

HipAccel.avails[HipAccel.name] = HipAccel
