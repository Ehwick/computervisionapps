# Computer vision apps
This project hosts two apps at once that use the tkinter and computer vision libraries. 

## Drawing app
 The first app is a drawing board that uses the mediapipe library, which allows us to create a hand mesh and follow our pointer-finger landmark. The webcam screen can be clicked to toggle the drawing function on and off, thereby allowing the user to draw with breaks in the paint.

i. Handmesh created using the mediapipe library and landmark recognition
![handmesh using mediapipe](/assets/handmesh.png)

ii. By toggling the drawing function on and off, the user can write and draw with breaks in between words and punctuation
![Drawing 'hi!'](/assets/hi.png)

## Drawing app
 The second app is a text recognition app that uses a Tesseract OCR engine, powered by TensorFlow, to allow for text recognition. Here the screen is flipped to allow for proper text display, a screenshot is taken, it is greycasted, and then Tesseract is use to return the attempt at recognition in the terminal.

i. The user can hold up text to the webcam and then press the 'Recognize Text' button. For optimal recognition, lighting should be appropriate and even, with minimum glare, and text should be clear and legible.
![Screenshot before text recognition](/assets/captured_image.jpg)

ii. The image is saved and the Tesseract OCR engine is used. In this case, the full text of this book's cover is read successfully! 
![Terminal response to screenshot](/assets/tensor.png)
