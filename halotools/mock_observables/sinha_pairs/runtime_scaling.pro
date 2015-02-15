function myfunct, X, P
  return, p[0] + p[1]*X^p[2]
end


compile_opt idl2, strictarrsubs

base_execstring = './DD cmassmock_AbM_swot_scatter0.10dex_All_V_PEAK_Zspace.ff f FILE FORMAT bins1 > xx'
second_files = ['random_AbM_swot_scatter0.10dex_All_V_PEAK_Zspace.ff', 'cmassmock_AbM_swot_scatter0.10dex_All_V_PEAK_Zspace.ff']
second_file_format = ['f', 'f']
legendstring = ['DR', 'DD']

base_execstring = './DD /home/sinham/code/andreas/halobias0_new/mine/gals_Mr19.ff f FILE FORMAT bins1 > xx'
second_files = ['/home/sinham/code/andreas/halobias0_new/mine/gals_Mr19.ff']
second_file_format = ['f']
legendstring = ['DD']
linestyle = [0]

nfiles = n_elements(second_files)
generate_eps = 0
do_nbin_timing = 0
timings_file = 'timings_Mr19'


if do_nbin_timing eq 0 then begin
   timings_file +=  '_r.txt'
endif else begin
   timings_file += '_nbin.txt'
endelse

;;;fixed things
ntries = 5
rpmin = 0.1

if do_nbin_timing eq 0 then begin
   rpmin = 0.1
   nbins = 20
   min_rpmax = 5.0
   max_rpmax = 50.0
   drpmax = 5.0
   nrps = fix((max_rpmax-min_rpmax)/drpmax) 
endif else begin
   min_nbins = 3
   max_nbins = 30
   dnbin = 1.0
   rpmax = 30.0
   nbin_tries = (max_nbins-min_nbins)/dnbin
endelse
   
if do_nbin_timing eq 0 then begin
   if n_elements(alltimings) ne nfiles*nrps*ntries then begin
      if n_elements(alltimings) eq 0 then begin
         alltimings = dblarr(nfiles, nrps, ntries)
      endif else begin
         newtimings = dblarr(nfiles, nrps, ntries)
         stop
      endelse
   endif
endif else begin
   if n_elements(alltimings) ne nfiles*nbin_tries*ntries then begin
      if n_elements(alltimings) eq 0 then begin
         alltimings = dblarr(nfiles, nbin_tries, ntries)
      endif else begin
         newtimings = dblarr(nfiles, nbin_tries, ntries)
         stop
      endelse
   endif
endelse

get_lun, lun
if (findfile(timings_file))[0] ne '' then begin
   print, 'FILE already exists'
   stop
endif


openw, lun, timings_file
printf, lun, '################################################################'
printf, lun, '# Type        time       iteration         rpmax        nbin    '
printf, lun, '#   s           d            l               d            l     '
printf, lun, '################################################################'
for ifile = 0, nfiles-1 do begin
   file2 = second_files[ifile]
   file2_format = second_file_format[ifile]
   type = legendstring[ifile]
   if do_nbin_timing eq 0 then begin
      for irp = 0, nrps-1 do begin
         this_nbin = nbins
         rpmax = min_rpmax + irp*drpmax
         execstring = './logbins ' + strn(rpmin) + " " + strn(rpmax) + " " + strn(this_nbin) + " > bins1"
         spawn, execstring
         execstring = str_replace(base_execstring, 'FILE', file2)
         execstring = str_replace(execstring, 'FORMAT', file2_format)
         for itries = 0, ntries-1 do begin
            if alltimings[ifile, irp, itries] eq 0.0 then begin
               t0 = systime(/seconds)
               spawn, execstring, dummy, dummy1
               ;; last_line = dummy1[-1]
               ;; yy = (strsplit(last_line, ' sec', /extract))[-1]
               t1 = systime(/seconds)
               alltimings[ifile, irp, itries] = t1-t0
               print, t1-t0, itries+1, rpmax, format = '("Time taken =  ",F8.3," for iteration # ",I3," with rpmax = ",F6.2, " Mpc/h")'
            endif
            time = alltimings[ifile, irp, itries]
            printf, lun, type, time, itries+1, rpmax, this_nbin, format = '(A10," ", F12.4," ",I12," ",F10.3," ",I10)'
            flush, lun
         endfor
         print, rpmax, format = '("Done with rpmax = ",F7.2," Mpc/h")'
      endfor
   endif else begin
      for ibin = 0, nbin_tries-1 do begin
         this_nbin = fix(min_nbins + ibin*dnbin)
         execstring = './logbins ' + strn(rpmin) + " " + strn(rpmax) + " " + strn(this_nbin) + " > bins1"
         spawn, execstring
         execstring = str_replace(base_execstring, 'FILE', file2)
         execstring = str_replace(execstring, 'FORMAT', file2_format)
         for itries = 0, ntries-1 do begin
            if alltimings[ifile, ibin, itries] eq 0.0 then begin
               t0 = systime(/seconds)
               spawn, execstring, dummy, dummy1
               t1 = systime(/seconds)
               alltimings[ifile, ibin, itries] = t1-t0
               print, t1-t0, itries+1, rpmax, this_nbin, format = '("Time taken =  ",F8.3," for iteration # ",I3," with nbin  = ",I3)'
            endif
            time = alltimings[ifile, ibin, itries]
            printf, lun, type, time, itries+1, rpmax,this_nbin, format = '(A10," ", F12.4," ",I12," ",F10.3," ",I10)'
            flush, lun
         endfor
         print, this_nbin, format = '("Done with ibin = ",I3)'
      endfor
   endelse
endfor
close, lun
free_lun, lun

if do_nbin_timing eq 0 then begin
   xdata = dindgen(nrps)*drpmax + min_rpmax
   if n_elements(timings) ne nfiles*nrps then begin
      timings = dblarr(nfiles, nrps)
      scatter = dblarr(nfiles, nrps)
   endif
endif else begin
   xdata = dindgen(nbin_tries)*dnbin + min_nbins
   if n_elements(timings) ne nfiles*nbin_tries then begin
      timings = dblarr(nfiles, nbin_tries)
      scatter = dblarr(nfiles, nbin_tries)
   endif
endelse

for ifile = 0, nfiles-1 do begin
   timings_data = reform(alltimings[ifile, *, *])
   timings[ifile, *] = mean(timings_data, dimension = 2)
   scatter[ifile, *] = stddev(timings_data, dimension = 2)
endfor

xrange = minmax(xdata)
yrange = minmax([timings-scatter, timings+scatter]) > 0.1
xticklen = 0.04
yticklen = 0.04
if do_nbin_timing eq 0 then begin
   xtitle = 'r [Mpc/h]'
endif else begin
   xtitle = 'nbins'
endelse

ytitle = 'runtime [seconds]'
position = [0.2, 0.2, 0.9, 0.9]



if generate_eps eq 0 then begin
   set_plot, 'x'
   thick = 3
   xthick = 4
   ythick = 4
   colors = ['red', 'blue']
   size = 800
   window, 0, xsize = size, ysize = size
endif else begin
   set_plot, 'ps'
   colors = ['red', 'dodgerblue']
   size = 8
   thick = 6
   xthick = 6
   ythick = 6
   device,  decomposed =  1
   !p.font =  1
   psfname = str_replace(timings_file, '.txt', '.eps')
   device,  filename =  psfname,  /encap,  /color,  /cmyk,  bits_per_pixel =  8,  xsize =  size,  ysize =  size,  /in,  /helvetica, /tt_font,  /bold,  $
            xoffset =  0,  yoffset =  0
   
endelse

parinfo =  replicate({value:0.D,  fixed:0,  limited:[0, 0],  $
                      limits:[0.D, 0]}, 3)
parinfo[*].limited[*] = 1
parinfo[0].limits = [0.001, 10.0]
parinfo[1].limits = [0.001, 100.0]
parinfo[2].limits = [0.001, 3.0]
parinfo[*].value = [0.2d, 2.0, 1.1]

plot, [0], xrange = xrange, yrange = yrange, /nodata, $
      xthick = xthick, ythick = ythick, xticklen = xticklen, yticklen = yticklen, $
      xtitle = xtitle, ytitle = ytitle, position = position, thick = thick, /ylog

for ifile = 0, nfiles-1 do begin
   this_timings = reform(timings[ifile, *])
   this_scatter = reform(scatter[ifile, *]) > 0.0001
   oploterror, xdata, this_timings, this_scatter, color = colors[ifile], line = linestyle[ifile], thick = thick, $
               errcolor = colors[ifile], errthick = thick

   result = mpfitfun('MYFUNCT', xdata[4:*], this_timings[4:*], this_scatter[4:*], parinfo = parinfo, yfit = yfit, /quiet)
   oplot, xdata[4:*], yfit, line = 2, thick = thick, color = cgcolor(colors[ifile])
   ;;textstring = 'O( rp!U' + string(result[2], format = '(F3.1)') +
   ;;'!N )'
   if do_nbin_timing eq 0 then begin
      textstring = tex2idl("$\propto$") + ' r!U' + string(result[2], format = '(F3.1)') + '!N'
   endif else begin
      textstring = tex2idl("$\propto$") + ' nbin!U' + string(result[2], format = '(F3.1)') + '!N'
   endelse
   xyouts,xdata[-5], yfit[-5], textstring, color = cgcolor(colors[ifile]), /data, alignment = 1
endfor
al_legend, legendstring, color = colors, line = linestyle, thick = thick, /top, /left, pspacing = 1.1, box = 0
plot, [0], xrange = xrange, yrange = yrange, /nodata, /ylog, $
      xthick = xthick, ythick = ythick, xticklen = xticklen, yticklen = yticklen, $
      xtitle = xtitle, ytitle = ytitle, position = position, thick = thick, /noerase


if !d.name eq 'PS' then begin
   device, /close
   set_plot, 'x'
endif

end
