for jpg in $(find train2017 -name *.jpg | sort -R | head -1000); do \
    cp ${jpg} calibrate_random/; \
done
