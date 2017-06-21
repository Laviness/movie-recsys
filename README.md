# movie-recsys
Its a movie recommendation system based on ML-100 database

Run under Python 3.5 with pandas, numpy, math, pandas and sklearn

The result is the RMSE of test set after several itrations.

```
Initialize end.The user number is:943,item number is:1682,the average score is:3.533063
Beginning to train the model......
Iteration 1 times,RMSE is : 0.983311
...
Iteration 18 times,RMSE is : 0.940043
Iteration 19 times,RMSE is : 0.940020
Iteration 20 times,RMSE is : 0.940007
Iteration 21 times,RMSE is : 0.940002
Iteration 22 times,RMSE is : 0.940003
Iteration finished!
```

The function generate will generate the full prediction score of all movies for each user. This step takes about an hour maybe because  I used the ```wirte``` function. You can comment line 156

```
s.generate()
```

if you only need to train the SVD model without generating all data.
