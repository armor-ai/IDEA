# input format: "visualize.sh app_name topic_number"
# run this file using 'visualize.sh music 8' for the app "music"
# $1 for the app name, $2 for the topic number
# 7778 is the port number for viewing the visualization, e.g., for localhost, we use localhost:8000 to observe the river.
# for python3, we run "python -m http.server" in the command line for visualization
python get_input.py $1 $2
ret=`python -c 'import sys; print("%i" % (sys.hexversion<0x03000000))'`
if [ $ret -eq 0 ]; then
    python -m http.server 7778
else
    python -m SimpleHTTPServer 7778
fi