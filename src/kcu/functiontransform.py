from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import (
    HasInputCol,
    HasOutputCol,
    Param,
    Params,
    TypeConverters,
)
import base64
import dill
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as F

class FunctionTransform(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    default_value: Param = Param(
        Params._dummy(),
        "default_value",
        "default_value",
        typeConverter=TypeConverters.toString,
    )

    parameter_value: Param = Param(
        Params._dummy(),
        "parameter_value",
        "parameter_value",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None,
                 default_value=None,
                 parameter_value=None):
        super(FunctionTransform, self).__init__()
        self.default_value = Param(self, "default_value", "unknown")
        self._setDefault(default_value=[])
        kwargs = self._input_kwargs
        print(kwargs)
        kwargs_ = dict()
        for key in kwargs:
            kwargs[key] = dill.dumps(kwargs[key]).decode(encoding="raw_unicode_escape")
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None,
                  default_value=None,
                  parameter_value=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setDefaultValue(self, value):
        return self._set(default_value=value)

    def getDefaultValue(self):
        return self.getOrDefault(self.default_value)
    
    def setParameterValue(self, value):
        return self._set(parameter_value=value)

    def getParameterValue(self):
        return self.getOrDefault(self.parameter_value)

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def _transform(self, dataset):
        specific_function = dill.loads(
                self.getDefaultValue().encode(encoding="raw_unicode_escape")
            )
        specific_params = dill.loads(
                self.getParameterValue().encode(encoding="raw_unicode_escape")
            )
        dset = specific_function(df=dataset, **specific_params)
        return dset