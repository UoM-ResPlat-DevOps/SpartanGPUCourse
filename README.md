# SpartanGPUCourse

An four hour training course that provides an introduction to programming on GPGPUs.

## Git

If you check this repository out be aware that it uses Git submodules to managt:

### fetch it all in one hit
`git clone --recursive https://github.com/UoM-ResPlat-DevOps/SpartanGPUCourse`

Or:

### take it step by step
`git clone https://github.com/UoM-ResPlat-DevOps/SpartanGPUCourse`
`git submodule init`
`git submodule update`

To regenerate the slides

.

To run it ensure that the java version installed is java 8:

`java -version`

should return something along the lines of java version "1.8.0_65".

java/javase/overview/java8-2100321.html`


`java -jar SlideExtractor.jar`

You should see something like the following fly by:

`Working on: ./Lessons/Lesson_1.md`
`Writing to: ./Presentation/Lesson_1.html`
`Writing to: ./Presentation/index.html`

## Folders

The directories that make up this project are as follows:

* Lessons - The lesson(s);
* Planning - The plan used to create the course;
* Resources - Resources for this particular run of the training.





