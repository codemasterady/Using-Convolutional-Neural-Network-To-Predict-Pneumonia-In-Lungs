# Introduction
One thing this pandemic has thought me was how reliant humanity is on technology from ordering products to assisting people in their professions. Due to the lack of hospital facilities around the world, I firmly believe that it is high time that we enhance our equipment to satisfy the mass amount of patients overflowing the hospitals every day. It is during times like these we realise that despite how advanced we become we are never prepared. Hence I am introducing my latest project in the advancement of **computer vision**, one that not only detects values to the highest precision but also generates results in the quickest. I am talking about the YOLO algorithm. For those of you who are unfamiliar don't look at me funny, we will discuss this in more detail. This isn't a solution to replacing doctors or the field of radiology, this algorithm is quite useless **for real purposes** * this algorithm just about shows how accurate I can derive the results for lungs with and without Pneumonia. That's right, this project is targeted in finding Pneumonia -like symptoms in the X-ray of the lungs and what it generates a value either 1 or 0 with **1 being has pneumonia** and **0 being doesn't have Pneumonia**. Okay now that we are clear that what this is intended for, let's look at the dataset shall we?


---

# The Dataset
As usual, the dataset is split into the training and the test set, but instead of being in an excel sheet, as all the data is images, the dataset is in a folder here is the architecture

|   File Path    | Description |
| ----------- | ----------- |
| train\NORMAL      | 1,342 images containing X-rays of normal lungs for testing     |
| train\PNEUMONIA   | 3,876 images containing X-rays of lungs with pneumonia for training        |
| test\NORMAL      | 234  images containing X-rays of normal lungs for validation     |
| test\PNEUMONIA   | 390 images containing X-rays of lungs with pneumonia for validation       |
