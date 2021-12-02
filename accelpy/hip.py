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

extern "C" int64_t accelpy_start(  \\
        int64_t * run_id, int64_t * device, int64_t * channel, \\
        int64_t * thread_x, int64_t * thread_y, int64_t * thread_z, \\
        int64_t * team_x, int64_t * team_y, int64_t * team_z, \\
        int64_t * assign_x, int64_t * assign_y, int64_t * assign_z) {{

    int64_t res;

    const dim3 TEAM_SIZE = dim3(*team_x, *team_y, *team_z);
    const dim3 MEMBER_SIZE = dim3(*thread_x, *thread_y, *thread_z);

    accelpy_kernel<<<TEAM_SIZE, MEMBER_SIZE>>>({launch_args});

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
class dev_{vartype} {{
public:
    {dtype} * data;
    int64_t * _attrs; // size, ndim, itemsize, shape, strides

    __device__ {dtype} & operator() ({oparg}) {{
        int64_t * s = &(_attrs[3+_attrs[1]]);
        return data[{offset}];
    }}

    __device__ {dtype} operator() ({oparg}) const {{
        int64_t * s = &(_attrs[3+_attrs[1]]);
        return data[{offset}];
    }}

    __device__ int size() {{
        return (int) _attrs[0];
    }}

    __device__ int ndim() {{
        return (int) _attrs[1];
    }}

    __device__ int itemsize() {{
        return (int) _attrs[2];
    }}

    __device__ int shape(int dim) {{
        return (int) _attrs[3+dim];
    }}

    __device__ int stride(int dim) {{
        return (int) _attrs[3+_attrs[1]+dim];
    }}

    __device__ int unravel_index(int tid, int dim) {{
        int q, r=tid, s;
        for (int i = 0; i < dim + 1; i++) {{
            s = stride(i);
            q = r / s;
            r = r % s;
        }}

        return q;
    }}
}};
"""

t_h2acopy = """
extern "C" int64_t {funcname}(int64_t * attrsize, int64_t * _attrs, void * data) {{
    int64_t res;

    hipMalloc((void **)&dev_{varname}.data, _attrs[0] * sizeof({dtype}));
    hipMalloc((void **)&dev_{varname}._attrs, (*attrsize) * sizeof(int64_t));

    hipMemcpyHtoD(dev_{varname}.data, data, _attrs[0] * sizeof({dtype}));
    hipMemcpyHtoD(dev_{varname}._attrs, _attrs, (*attrsize) * sizeof(int64_t));

    res = 0;

    return res;
}}
"""

t_h2amalloc = """
extern "C" int64_t {funcname}(int64_t * attrsize, int64_t * _attrs, void * data) {{
    int64_t res;

    hipMalloc((void **)&dev_{varname}.data, _attrs[0] * sizeof({dtype}));
    hipMalloc((void **)&dev_{varname}._attrs, (*attrsize) * sizeof(int64_t));

    hipMemcpyHtoD(dev_{varname}._attrs, _attrs, (*attrsize) * sizeof(int64_t));

    res = 0;

    return res;
}}
"""

t_a2hcopy = """
extern "C" int64_t {funcname}(int64_t * attrsize, int64_t * _attrs, void * data) {{
    int64_t res;

    hipMemcpyDtoH(data, dev_{varname}.data, _attrs[0] * sizeof({dtype}));

    res = 0;

    return res;
}}
"""

t_testfunc = """
__global__ void accelpy_test_kernel(dev_{vartype_in} {varin}, dev_{vartype_out} {varout}){{

    int ACCELPY_WORKER_ID0 = blockIdx.x * blockDim.x + threadIdx.x;
    int ACCELPY_WORKER_ID1 = blockIdx.y * blockDim.y + threadIdx.y;
    int ACCELPY_WORKER_ID2 = blockIdx.z * blockDim.z + threadIdx.z;

    int id = ACCELPY_WORKER_ID0;
    if(id < {varin}.size()) {varout}(id) = {varin}(id);

}}

extern "C" int64_t accelpy_test_run() {{
    int64_t res;

    const dim3 TEAM_SIZE = dim3(1);
    const dim3 MEMBER_SIZE = dim3({nworkers});

    accelpy_test_kernel<<<TEAM_SIZE, MEMBER_SIZE>>>(dev_{varin}, dev_{varout});

    res = 0;

    return res;
}}
"""

t_kernel = """
__global__ void accelpy_kernel({kernel_args}){{

    int ACCELPY_WORKER_ID0 = blockIdx.x * blockDim.x + threadIdx.x;
    int ACCELPY_WORKER_ID1 = blockIdx.y * blockDim.y + threadIdx.y;
    int ACCELPY_WORKER_ID2 = blockIdx.z * blockDim.z + threadIdx.z;

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
            vartype = self.get_vartype(input)
            dtype = self.get_dtype(input)
            funcname = self.getname_h2acopy(input)

            out.append("dev_%s dev_%s = dev_%s();" % (vartype, input["curname"], vartype))
            out.append(t_h2acopy.format(funcname=funcname, varname=input["curname"], dtype=dtype))

        for output in outputs:
            vartype = self.get_vartype(output)
            dtype = self.get_dtype(output)
            funcname = self.getname_h2amalloc(output)

            out.append("dev_%s dev_%s = dev_%s();" % (vartype, output["curname"], vartype))
            out.append(t_h2amalloc.format(funcname=funcname, varname=output["curname"],
                        dtype=dtype))

        for output in outputs:
            funcname = self.getname_a2hcopy(output)
            dtype = self.get_dtype(output)

            out.append(t_a2hcopy.format(funcname=funcname, varname=output["curname"],
                        dtype=dtype))

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

        out.append("dev_%s dev_%s = dev_%s();" % (vartype_in, input["curname"], vartype_in))
        out.append("dev_%s dev_%s = dev_%s();" % (vartype_out, output["curname"], vartype_out))

        out.append(t_h2acopy.format(funcname=funcname_in, varname=input["curname"], dtype=dtype_in))
        out.append(t_h2amalloc.format(funcname=funcname_out, varname=output["curname"], dtype=dtype_out))
        out.append(t_a2hcopy.format(funcname=funcname_a2h, varname=output["curname"], dtype=dtype_out))

        out.append(t_testfunc.format(varin=input["curname"], varout=output["curname"],
                    vartype_in=vartype_in, vartype_out=vartype_out, nworkers=str(input["data"].size)))

        return "\n".join(out)

    def gen_kernel(self, inputs, outputs):

        args = []
        order =  self._order.get_section(self.name)

        for data in inputs+outputs:
            vartype = self.get_vartype(data)
            args.append("dev_%s %s" % (vartype, data["curname"]))

        return t_kernel.format(order="\n".join(order.body), kernel_args=", ".join(args))

    def gen_code(self, inputs, outputs):

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

HipAccel.avails[HipAccel.name] = HipAccel
