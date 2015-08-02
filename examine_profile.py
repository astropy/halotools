import pstats
prof = pstats.Stats('profile_abz.prof')
prof.strip_dirs()
prof.sort_stats('time').print_stats()
prof.sort_stats('time').print_callers()
