# pytracking
1. Put the model into mfDiMP/pytracking/DiMP_nets/debug_lichao_comb23_2

2. in mfDiMP/pytracking/evaluation/local.py
You should change the path settings.network_path as with your absolute path '....../mfDiMP/pytracking/DiMP_nets'

3. the tracker file for vot-toolkit is in mfDiMP/pytracking/track_vot/tracker_mfDiMP.m
you can move 'tracker_mfDiMP.m' to your vot-toolkit workspace.
*Note: open tracker_mfDiMP.m, there are some path and link_path should be changed, you may refer to it to change as your environment

4. in mfDiMP/pytracking/track_vot/run_debug_lichao_comb23_2.py, 
you should also change the 'sys.path.append('/home/lichao/projects/pytracking_lichao')' to your path pointing to mfDiMP

