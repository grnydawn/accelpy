    @classmethod
    def _dimension(cls, arg, attrspec):

        aid = arg["id"]

        if aid in attrspec and "dimension" in attrspec[aid]:
            return attrspec[aid]["dimension"]
                
        return ", ".join([str(s) for s in arg["data"].shape])

    @classmethod
    def _modvardefs(cls, arg):

        dim =  ", ".join([":" for s in arg["data"].shape])

        if dim:
            return "%s, DIMENSION(%s), POINTER :: %s" % (get_f_dtype(arg),
                dim, arg["curname"])

        else:
            return "%s, POINTER :: %s" % (get_f_dtype(arg), arg["curname"])

    @classmethod
    def _entervardefs(cls, arg, intent, attrspec={}):

        dim = cls._dimension(arg, attrspec)

        if dim:
            return "%s, DIMENSION(%s), INTENT(%s), TARGET :: %s" % (get_f_dtype(arg),
                dim, intent, "l"+arg["curname"])

        else:
            return "%s, INTENT(IN) :: %s" % (get_f_dtype(arg), arg["curname"])

    @classmethod
    def _kernelvardefs(cls, arg, intent, attrspec={}):

        curname = arg["curname"]

        if curname in attrspec and "dimension" in attrspec[curname]:
            return attrspec[curname]["dimension"]
                
        dim = ", ".join([str(s) for s in arg["data"].shape])

        if dim:
            return "%s, DIMENSION(%s), INTENT(%s), TARGET :: %s" % (get_f_dtype(arg),
                dim, intent, "l"+arg["curname"])

        else:
            return "%s, INTENT(IN) :: %s" % (get_f_dtype(arg), arg["curname"])

    @classmethod
    def gen_datafile(cls, modname, filename, runid, workdir, copyinout,
                        copyin, copyout, alloc, attr):

        datapath = os.path.join(workdir, filename)

        dataparams = {"runid": str(runid), "datamodname": modname}

        modvardefs = []
        modvars = []

        enterargs = []
        entervardefs = []
        enterassign = []

        for cio in copyinout:
            cioname = cio["curname"]

            modvars.append(cioname)
            enterargs.append("l"+cioname)
            entervardefs.append(cls._entervardefs(cio, "INOUT", attrspec=attr))
            modvardefs.append(cls._modvardefs(cio))
            enterassign.append("%s => %s" % (cioname, "l"+cioname))

        for ci in copyin:
            ciname = ci["curname"]

            modvars.append(ciname)
            enterargs.append("l"+ciname)
            entervardefs.append(cls._entervardefs(ci, "INOUT", attrspec=attr))
            modvardefs.append(cls._modvardefs(ci))
            enterassign.append("%s => %s" % (ciname, "l"+ciname))

        for co in copyout:
            coname = co["curname"]

            modvars.append(coname)
            enterargs.append("l"+coname)
            entervardefs.append(cls._entervardefs(co, "INOUT", attrspec=attr))
            modvardefs.append(cls._modvardefs(co))
            enterassign.append("%s => %s" % (coname, "l"+coname))

        for al in alloc:
            alname = al["curname"]

            modvars.append(alname)
            enterargs.append("l"+alname)
            entervardefs.append(cls._entervardefs(al, "INOUT", attrspec=attr))
            modvardefs.append(cls._modvardefs(al))
            enterassign.append("%s => %s" % (alname, "l"+alname))

        dataparams["modvardefs"] = "\n".join(modvardefs)
        dataparams["modvars"] = ", ".join(modvars)
        dataparams["enterargs"] = ", ".join(enterargs)
        dataparams["entervardefs"] = "\n".join(entervardefs)
        dataparams["enterdirective"] = ""
        dataparams["enterassign"] = "\n".join(enterassign)
        dataparams["exitdirective"] = ""

        with open(datapath, "w") as fdata:
            fdata.write(moddatasrc.format(**dataparams))

        return datapath

    @classmethod
    def gen_kernelfile(cls, knlhash, dmodname, runid, section, workdir, localvars, modvars):

        kernelpath = os.path.join(workdir, "K%s%s" % (knlhash[2:], cls.srcext))

        kernelparams = {"runid": str(runid)}
        kernelargs = []
        uonlyvars = []
        kernelvardefs = []

        for old, newdata in modvars:
            uonlyvars.append("USE %s, ONLY : %s => %s" % (dmodname, newdata["curname"], old))

        attrspec = section.kwargs.get("attrspec", {})

        for lvar in localvars:

            kernelargs.append(lvar["curname"])
            kernelvardefs.append(cls._kernelvardefs(lvar, "INOUT", attrspec=attrspec))

        kernelparams["kernelmodname"] = "MOD%s" % knlhash[2:].upper()
        kernelparams["kernelargs"] = ", ".join(kernelargs)
        kernelparams["useonlyvars"] = "\n".join(uonlyvars)
        kernelparams["kernelvardefs"] = "\n".join(kernelvardefs)
        kernelparams["kernelbody"] = "\n".join(section.body)

        with open(kernelpath, "w") as fkernel:
            fkernel.write(modkernelsrc.format(**kernelparams))

        #import pdb; pdb.set_trace()
        return kernelpath
