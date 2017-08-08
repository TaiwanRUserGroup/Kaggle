## Pretrained Model

run `cat models.a* > models.pickle`, then load the pretrained models with `pickle` package in Python3.

You'll get a dictionary with byte string of the url as key and another dictionary as value which has model lags as key and model information as value.

Ex:
```
# key
b"A'N'D_zh.wikipedia.org_all-access_spider",
# value
 {1: {
      'loss': 61.12005165109715,
      'success': True,
      'x': array([  0.43242928,  16.54056709])
      },
  3: {
      'loss': 60.448980413384412,
      'success': True,
      'x': array([ 0.10801921,  0.33119062,  0.35785994,  1.18146586])
     },
  5: {
      'loss': 59.785036307518233,
      'success': True,
      'x': array([ 0.12179051,  0.18429717,  0.14235019,  0.22654872,  0.14040482, 0.10531036])
     },
  7: {
      'loss': 59.034570945715906,
      'success': True,
      'x': array([ 0.04536065,  0.07703323,  0.09787072,  0.01195619,  0.07330605,
                   0.15990223,  0.28351953,  0.92945804])
     }
 }
```
