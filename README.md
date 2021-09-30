# apneAI
Hackxplore 2021 Machine Learning submission 
# Inspiration
We all sleep! Some of us sleep well, and some don’t! This quality rules a large chunk of our days. We can be either fresh and active or tired and drowsy during the day. Many workplace and road accidents are associated with drowsiness during the active times of the day. A prevalent disorder that causes this is Obstructive Apnea.

# What is sleep apnea?
It’s a disorder wherein your airways collapse during sleep and you can’t breathe for a few seconds while in deep sleep. It sometimes wakes you up, or restarts your sleep to a lighter stage, disrupting your brain’s recovery during the night. In the long term, it has been demonstrated that apnea can increase your chances of Alzheimer’s Disease and brain cancer. Up to 10% of the population can suffer from this disorder; that is around 3.5 million people in Canada and 800 million people worldwide! Most cases remain undiagnosed because nobody is awake to watch you sleep (obviously!).

# What it does
With the advent of wearable technologies these days, it makes it easy for us to gain a window into our health when we previously couldn’t. An example of that is, we can use FDA-registered devices like Muse S (which is actually made by UofT alumni! But we digress…) or the Dream headband to record our brain activity during sleep. We designed an algorithm to detect sleep apnea while you’re asleep and nobody is watching you!

# How we built it
We acquired sleep recording datasets online (from the University College Dublin, at physionet.org/content/ucddb/1.0.0/), containing sleep recordings from 10 individuals, which is recorded voltage across time. We then extracted the power spectra of these recordings using the Fast Fourier Transform (FFT) in 5 second windows, demonstrating how strong each frequency was observed across time in the EEG. We used these frequencies as the feature to run our learning algorithm. Even without an accelerator, you can complete these computations in a few hundred milliseconds.

We then trained a Gaussian Naive Bayes Classifier to differentiate between apnea and non apnea events, as this is a very fast method that could run in real-time.

# Challenges we ran into
Time was difficult to juggle, we wanted to try different machine learning models and see which ones fit best (and have the most research on sleep apnea). Unfortunately, we did not have time to test more than 2 as cleaning up the data and training the data took longer than anticipated. In terms of the visualization, it was really challenging to portray the entirety of the dataset as it had six different layers after we already preprocessed it to remove some of the noise. It can be very difficult to remove all the different types of noise without accounting for the loss of potentially usable data for the detection of apnea. So finding an applicable plot figure that is able to demonstrate all the information we wanted the user to be able to see was quite difficult.

# Accomplishments that we're proud of
We were very proud of our overall pipeline design and our ability to quickly pull out apnea vs non-apnea signatures from our dataset. We were also very happy we could bring all of this together during the quick turnaround demanded by the hackathon, and that we now have the basis for a platform that we can continue to improve in the future!

# What we learned
Firstly, we were very happy to find that publicly available, annotated sleep data is available that can allow us to build a machine learning model for sleep apnea. With the right tools, it should be possible to continue applying our model to new datasets that emerge as an alternative to the human annotation of sleep data, thus contributing to these publicly-available repositories of data as well.

The spirit of teamwork allowed us to accelerate the process even to the place it is in an incredibly short period of time, and we will all cherish the experience for our careers.

# What's next for ApneAI
We plan to build a mobile app, which syncs with a wearable EEG device (or ECG heart monitor) We will use patient data in real time, in our ML pipeline Alerts them when it detects an apnea event We plan to implement this on a GPU-accelerated embedded system, such as the Jetson nano, to provide a complete suite of machine learning solutions for diagnosing sleep disorders and improving the sleep states.