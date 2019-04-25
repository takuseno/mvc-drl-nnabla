# mvc-dr-nnabla
NNabla implementation with [mvc-drl](https://github.com/takuseno/mvc-drl).

## installation
### nvidia-docker
```
$ ./scripts/build.sh
```

### manual
```
$ pip install -r requirements.txt
$ pip install nnabla
# if you run example scripts
$ pip install pybullet roboschool
```

If you use GPU, see [here](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html).

## algorithms
For academic usage, we provide baseline implementations that you might need to compare.

- [ ] Proximal Policy Optimization
- [x] Deep Deterministic Policy Gradients
- [ ] Soft Actor-Critic


