## Package project to run on ML Engine

To package the MiniMNIST project run:

```bash
python setup.py sdist
```

You'll see that a `dist` folder is created and inside that there is a **tar.gz** file. That file is you'll use to submit your code to ML Engine.