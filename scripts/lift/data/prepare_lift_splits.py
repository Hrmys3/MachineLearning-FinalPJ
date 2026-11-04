"""运行入口；实现位于 mlpj.lift.data.prepare_lift_splits。"""
import runpy

if __name__ == "__main__":
    runpy.run_module("mlpj.lift.data.prepare_lift_splits", run_name="__main__")
