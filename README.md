# Instant Vital Checkup (IVC)
Patients get quick and fair treatment, hospitals save on human power

### Technology and Frameworks used
&nbsp;&nbsp;&nbsp;&nbsp; - `Django`<br>
&nbsp;&nbsp;&nbsp;&nbsp; - `OpenCV`<br>
&nbsp;&nbsp;&nbsp;&nbsp; - `Tensorflow`<br>
&nbsp;&nbsp;&nbsp;&nbsp; - `HTML`<br>
&nbsp;&nbsp;&nbsp;&nbsp; - `CSS`<br>
&nbsp;&nbsp;&nbsp;&nbsp; - `Bootstrap`<br>

### Problem Statement

Each day, thousands die due to medical negligence and delay in attending critical patients. Most of the time this is due to delay in completing pre-surgical procedures because of lack of nurses and a big volume of patients.

You might have experienced the same. Remember going to see the doctor for a common cold and being told you can only see the doctor after completing the pre-requisite checks like height, weight, bmi, pulse, temperature, etc? Sometimes these procedures take more time than the actual appointment, not to mention the costs involved in getting these procedures done.

We aim at automating such procedures using Computer Vision (OpenCV) and Machine Learning (ML) technologies.

### Idea

  With the power of Machine Learning, Computer Vision and a few well placed sensors, we can entirely eliminate the need for a human attendee in the pre-surgical procedures, giving us quicker and more reliable results. This would eliminate the otherwise time and human intensive work of measuring vitals of a person. 
  
  Using well placed cameras, we will be able to monitor the body temperature, pulse and get a good estimate of the weight and height of the person to get an idea of their Body Mass Index(BMI). This would be done while the person is filling out the registration form, thus ensuring quick treatment for the affected.

### Working

  • IVC has a special setup where when the person is at the marked spot, the camera takes his/her picture, and using trigonometric ratios, the person's height, arm length, waist breadth, and leg length are determined. Presently two parameters need to be hard coded(height and distance of camera).
  We also made an approximate guess of waist circumference using frontal waist breadth.
  
  • Using a pre-trained open-source deep neural network and computer vision which takes video feed as input, we estimated a person's age and gender.
  
  • Using the parameters estimated in the previous slides i.e. height, arm length, leg length, gender, age, and waist circumference, we trained a random forest model to estimate the weight. The model was able to deliver an estimated accuracy of 98.6%.
  
  • To get your pulse measured the person has to walk towards the camera.
  The camera takes a zoomed video of the forehead, using computer vision we are able to estimate the pulse of a person by measuring the change in skin colour of the person’s forehead.


### Web Interface
<img width="964" alt="image" src="https://github.com/Aman281201/Instant-Vital-Checkup/assets/74315059/0968a937-8b0e-4dda-8bc7-d29fd2415adf">

<img width="1057" alt="image" src="https://github.com/Aman281201/Instant-Vital-Checkup/assets/74315059/82bd4c69-1f8a-429b-98a3-49eec71734a2">

<img width="854" alt="image" src="https://github.com/Aman281201/Instant-Vital-Checkup/assets/74315059/a9d2f2f6-1273-4289-898f-cd15b8b78b2e">



