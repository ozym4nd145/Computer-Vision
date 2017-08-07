# Assignment 1: Lamp Post measurement
#### Submitted by: Suyash Agrawal , 2015CS10262

### Basic Assumptions
* We require the field of view of the camera. Either this can be given in specifications itself or this can be easily calculated by placing an object of known length at a known distance from camera (this gives us the angle subtended ) and dividing the angle with the ratio of the length of object in image by image height. Let us denote the field of view of camera by $\Omega$.
* We assume the camera to be kept on the ground and thus field of vision will be reduced by half.
* The object is sufficiently far away that whole of it is visible in the image.

![Diagram](http://i.imgur.com/n8Pof2bg.jpg)

### Procedure for vertical height measurement
* Take a picture from unknown distance $l$ of the pole. Let the real height of pole be $h$ (vertically from ground). Let picture height be $w$ and height of pole in picture be $p$.
* Move $d$ distance towards the pole and take a picture. Let the new height of pole in picture be $p'$ .
* Now using equation $iv$ we can measure angle $\theta$ and using equation $iii$ we can measure angle $\theta'$ .
* Then using equation $v$ we measure the vertical height $h$ of pole from ground.
![Equations](http://i.imgur.com/kVh5jWn.jpg)
### Procedure for angle inclination measurement
* First measure the actual length of pole ( $L$ ) in image and multiplying it by a factor $h/p$.
* Then calculate the inclination from vertical using : $\theta_a$ = $\cos^{-1}(L/h)$ 
