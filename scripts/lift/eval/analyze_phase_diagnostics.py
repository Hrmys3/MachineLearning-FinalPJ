"""阶段诊断只读分析入口。"""
import runpy

if __name__ == '__main__':
    runpy.run_module('mlpj.lift.eval.analyze_phase_diagnostics', run_name='__main__')
