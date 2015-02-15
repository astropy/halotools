pro write_fortran,   lun,   nelements,   size,   var
  compile_opt idl2,   strictarrsubs
  bytes =  long(size*nelements)
  writeu,   lun,   bytes
  writeu,   lun,   var
  writeu,   lun,   bytes
end


compile_opt idl2, strictarrsubs

;;; Input params
file = 'random_AbM_swot_scatter0.10dex_All_V_PEAK_Zspace.txt'
outfile = 'random_AbM_swot_scatter0.10dex_All_V_PEAK_Zspace.ff'

file = 'cmassmock_AbM_swot_scatter0.10dex_All_V_PEAK_Zspace.txt'
outfile = 'cmassmock_AbM_swot_scatter0.10dex_All_V_PEAK_Zspace.ff'
double_precision = 1

;;; Try to guess the output file
if n_elements(outfile) eq 0 then begin
   subparts =  StrSplit(file,  ".",  /Extract)
   numsubParts =  N_Elements(subparts)
   if numsubparts gt 1 then begin
      extension = subparts[numsubParts-1]   
      outfile = str_replace(file, extension, '.ff')
   endif else begin
      print, 'ERROR: Can not figure out file extension - please supply outfile'
   endelse
endif


if n_elements(x) eq 0 then begin
   if double_precision eq 1 then begin
      readcol, file, x, y, z, format = 'D,D,D'
      size = 8
   endif else begin
      readcol, file, x, y, z, format = 'F,F,F'
      size = 4
   endelse
endif

rcube = max([x, y, z])
N = n_elements(x)
znow = 0.0

idat = lonarr(5)
fdat = fltarr(9)

idat[0] = fix(rcube)
idat[1] = N
fdat[0] = rcube

openw, lun, outfile, /get_lun
write_fortran, lun, 1, 20, idat
write_fortran, lun, 1, 36, fdat
write_fortran, lun, 1, 4, znow
write_fortran, lun, N, size, x
write_fortran, lun, N, size, y
write_fortran, lun, N, size, z
close, lun
free_lun, lun

end
