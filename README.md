# Long Exposure
This is our CSCI1290 Computational Photography final project by Mandy He, Helen Huang, and Tiger Lamlertpraserkul. 

We reimplemented the paper, *Computational Long Exposure Mobile Photography* by Tabellion et al. The original paper can be found [here](https://arxiv.org/pdf/2308.01379.pdf).

## Installation
To run this code, please install the packages defined in `requirements.txt` by running the following command:

```
$ pip install -r requirements.txt
```

## Running the Code
To run the full pipeline, run the following command at the root of the project:

```
python3 pipeline.py
```

### Changing Options
You can change the several options in `pipeline.py`:
- To change between using `cv2` or `RAFT` for generating the optical flow maps, change the `method` variable on `line 236` to either of the following options:
  - `cv2`
  - `raft`
- To used cached images/data, pass in the `from_cache=True` argument to the relevant function calls in `pipeline()`
