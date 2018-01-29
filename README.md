# Tensorflow Classification Framework


## New releases

Use the `release.sh` bash script to release new versions:

```bash
# usage
# ./release.sh <semantic version>
# example 
./release.sh 1.0 
```

**IMPORTANT**: Follow [semantic versioning specification](https://semver.org/) when choosing the version name.

The example above will:
 
 * check if the tree is clean
 * create the branch `release-1.0`
 * change the `__version__` variable in `__init__.py`
 * commit the changes
 * create and push a new tag `1.0`
 
 The creation of this new tag will trigger a new bitbucket pipeline that will create the distribution package and 
 copy it to the bucket: `gs://tfcf/releases`. See `bitbucket-pipelines.yml` for more details.
 
 ## Build and install
 ```bash
 python setup.py sdist
 
 sudo pip install dist/tf_image_classification-2.1.0.tar.gz --upgrade
 ```
  