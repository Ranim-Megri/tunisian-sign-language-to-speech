# Learning notes

## What we learned this week

- We downloaded the Kaggle dataset.
- The dataset has 57 sign folders and 4423 images.
- We selected 13 sign folders for this training project.
- The selected folders contain 852 pictures before landmark extraction.
- MediaPipe 1.1.1 uses the Tasks API.
- Each detected picture produces 129 landmark features.
- Pandas could not load in this Windows environment because a compiled DLL was blocked.
- The landmark notebook now uses Python's built-in CSV module for its output files.

## Problems and how we fixed them

- The data folder was empty. We downloaded the dataset with `kagglehub` and copied it to `data/raw/`.
- The first MediaPipe API was not available. We changed the notebook to use `HandLandmarker`.
- Pandas failed to load. We removed Pandas from the landmark extraction notebook.

## Questions for the teacher

- Which person can validate the TunSL signs and draft sentences?
- Which signs should be removed if their hand detection rate is below 70 percent?
- Which model score should we accept before testing the gadget camera?

## Week 1

- 

## Week 2

- 

## Week 3

- 

## Week 4

- 

## Week 5

- 

## Week 6

- 
