## What does QTcAnalyser do?
This tool is used to visualize latencies between Qwave onset and Twave offest of recorded ECG data. Raw input data is filtered and pre-processed using NeuroKit2 prior to detection of waveform characteristics. 



![alt text](https://github.com/Ahomagai/QTcAnalyzer/blob/main/QTcDashboard_demo.png)



## **Getting started**
- First, download the files of the repository, then in a terminal run the following sequentially:

  - ```python -m venv .venv```
  - ```.venv\Scripts\activate```
  - ```pip install -r requirements.txt```
  - ```python qt_dash_app.py```

Then the program will run in your local web browser, to view the output, you can go to:
```http://127.0.0.1:8050/```

## **If the above does not work you can also:**

Install python, and install the following packages using pip (copy paste these lines in your terminal):
- ```pip install dash```
- ```pip install pandas```
- ```pip install numpy```
- ```pip install neurokit2```
- ```pip install plotly```

Then double click qt_dash_app.py to open the program 

And then drag and drop or click the button to upload your ECG file.

## **Important to know**

The visualization runs in your browser locally. If you want to run the program again, simply close the browser and run qt_dash_app.py again and upload your file as necessary. 

The graphs are both interactable and should be carefully examined to see if there are any large variabilites in detected Q_onset and T_offset periods. 
