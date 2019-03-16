if [ ! -d "cars_test" ]; then
  wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
  tar -zxvf cars_test.tgz
fi

if [ ! -d "cars_train" ]; then
  wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
  tar -zxvf cars_train.tgz
fi

if [ ! -d "devkit" ]; then
 wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
 tar -zxvf devkit.tgz
fi
