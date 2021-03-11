## the module that takes in intervals_df and looks in the transcript to assign
## the raw text that matches that gesture, and does further semantic mapping of that text
## and outputs the resulting csv into intervals_with_semantics_df.csv.


## reads intervals_df into pd df
## Downloads transcript from google cloud
## Does some logic over whether that transcript is old transcription or a new one.
## matches timestamps of transcript to timestamps of gestures
## Does semantic analysis of the transcript segment that matches that gesture.
## outputs result into intervals_with_semantics_df.csv
## which contains fp to audio, video, and npz (keypoint) data.
# Thus outputs entire