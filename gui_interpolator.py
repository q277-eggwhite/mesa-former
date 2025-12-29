#!/usr/bin/env python3
"""
恒星质量预测插值软件 - GUI版本

功能：使用训练好的Transformer模型预测恒星质量
输入：恒星光度(logL)、有效温度(logTeff)、金属丰度(FeH)
输出：预测的恒星质量
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import pandas as pd
import os

# 导入模型加载和预测函数
from interpolator import load_model, predict_mass, load_standardization_params

class MassInterpolatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("恒星质量预测插值软件")
        self.root.geometry("500x500")
        self.root.resizable(False, False)
        
        # 模型和标准化参数相关
        self.model = None
        self.device = "cpu"
        self.model_loaded = False
        self.standardization_params = None
        
        # 创建界面
        self.create_widgets()
        
        # 在后台加载模型和标准化参数
        self.load_model_thread = threading.Thread(target=self.load_model_bg)
        self.load_model_thread.daemon = True
        self.load_model_thread.start()
    
    def create_widgets(self):
        # 标题
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(title_frame, text="恒星质量预测插值软件", 
                               font=("Arial", 16, "bold"))
        title_label.pack(anchor=tk.CENTER)
        
        # 模型状态
        status_frame = ttk.Frame(self.root, padding="5")
        status_frame.pack(fill=tk.X)
        
        self.status_var = tk.StringVar()
        self.status_var.set("模型加载中...")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                foreground="blue", font=("Arial", 10))
        status_label.pack(anchor=tk.CENTER)
        
        # 参数输入区域
        input_frame = ttk.LabelFrame(self.root, text="输入参数", padding="15")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # logL
        ttk.Label(input_frame, text="logL (恒星光度的对数):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.logL_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.logL_var, width=20).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # logTeff
        ttk.Label(input_frame, text="logTeff (有效温度的对数):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.logTeff_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.logTeff_var, width=20).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # FeH
        ttk.Label(input_frame, text="FeH (金属丰度):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.FeH_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.FeH_var, width=20).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 示例按钮
        ttk.Button(input_frame, text="使用示例参数", command=self.load_example).grid(row=4, column=0, columnspan=2, 
                                                                                pady=10)
        
        # 预测按钮
        self.predict_btn = ttk.Button(input_frame, text="预测恒星质量", command=self.predict_mass, 
                                      state=tk.DISABLED)
        self.predict_btn.grid(row=5, column=0, columnspan=2, pady=10)
        
        # 结果区域
        result_frame = ttk.LabelFrame(self.root, text="预测结果", padding="15")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.result_var = tk.StringVar()
        self.result_var.set("预测结果将显示在这里")
        result_label = ttk.Label(result_frame, textvariable=self.result_var, 
                                font=("Arial", 12), justify=tk.CENTER)
        result_label.pack(fill=tk.BOTH, expand=True)
        
        # 批量处理按钮
        batch_frame = ttk.Frame(self.root, padding="10")
        batch_frame.pack(fill=tk.X)
        
        ttk.Button(batch_frame, text="批量处理CSV文件", command=self.batch_process).pack(anchor=tk.CENTER)
        
        # 状态栏
        self.progress_var = tk.StringVar()
        self.progress_var.set("")
        progress_label = ttk.Label(self.root, textvariable=self.progress_var, 
                                  font=("Arial", 8), foreground="gray")
        progress_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model_bg(self):
        """后台加载模型和标准化参数"""
        try:
            # 获取脚本所在目录的绝对路径
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 加载标准化参数
            self.progress_var.set("正在加载标准化参数...")
            # 计算标准化参数文件的绝对路径
            params_path = os.path.join(script_dir, "data", "standardization_params.json")
            self.standardization_params = load_standardization_params(params_path)
            
            # 加载模型
            # 计算模型文件的绝对路径
            model_path = os.path.join(script_dir, "models", "best_model.pth")
            self.progress_var.set(f"正在加载模型: {model_path}...")
            self.model = load_model(model_path, self.device)
            
            self.model_loaded = True
            self.status_var.set("模型加载完成，可以开始预测")
            self.predict_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.status_var.set(f"模型加载失败: {e}")
            self.progress_var.set("")
    
    def load_example(self):
        """加载示例参数"""
        self.logL_var.set("3.5")
        self.logTeff_var.set("4.1")
        self.FeH_var.set("-0.2")
    
    def predict_mass(self):
        """预测恒星质量"""
        if not self.model_loaded:
            messagebox.showerror("错误", "模型尚未加载完成，请稍候")
            return
        
        try:
            # 获取输入值
            logL = float(self.logL_var.get())
            logTeff = float(self.logTeff_var.get())
            FeH = float(self.FeH_var.get())
            
            # 更新状态
            self.progress_var.set("正在预测中...")
            self.root.update_idletasks()
            
            # 进行预测 - 使用正确的参数顺序并传递标准化参数
            predicted_mass = predict_mass(self.model, logL, logTeff, FeH, self.device, self.standardization_params)
            
            # 显示结果
            result_text = f"预测的恒星质量: {predicted_mass:.4f} 太阳质量\n"
            result_text += f"输入参数:\n"
            result_text += f"logL: {logL:.2f}\n"
            result_text += f"logTeff: {logTeff:.2f}\n"
            result_text += f"FeH: {FeH:.2f}\n"
            self.result_var.set(result_text)
            self.progress_var.set("")
            
        except ValueError as e:
            messagebox.showerror("输入错误", f"参数输入有误: {e}")
            self.progress_var.set("")
        except Exception as e:
            messagebox.showerror("预测错误", f"预测过程中出错: {e}")
            self.progress_var.set("")
    
    def batch_process(self):
        """批量处理CSV文件"""
        if not self.model_loaded:
            messagebox.showerror("错误", "模型尚未加载完成，请稍候")
            return
        
        # 选择输入文件
        input_file = filedialog.askopenfilename(
            title="选择输入CSV文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if not input_file:
            return
        
        # 选择输出文件
        output_file = filedialog.asksaveasfilename(
            title="保存预测结果",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if not output_file:
            return
        
        # 后台处理文件
        def process_file_bg():
            try:
                self.progress_var.set(f"正在处理文件: {os.path.basename(input_file)}...")
                self.root.update_idletasks()
                
                # 读取输入文件
                df = pd.read_csv(input_file)
                
                # 检查必要的列
                required_columns = ['logL', 'logTeff', 'FeH']
                for col in required_columns:
                    if col not in df.columns:
                        messagebox.showerror("错误", f"输入文件缺少必要的列: {col}")
                        self.progress_var.set("")
                        return
                
                # 批量预测 - 使用正确的参数顺序并传递标准化参数
                predictions = []
                for _, row in df.iterrows():
                    mass = predict_mass(self.model, row['logL'], row['logTeff'], 
                                      row['FeH'], self.device, self.standardization_params)
                    predictions.append(mass)
                
                # 保存结果
                df['predicted_mass'] = predictions
                df.to_csv(output_file, index=False)
                
                self.progress_var.set("")
                messagebox.showinfo("成功", f"预测完成！结果已保存到:\n{output_file}")
                
            except Exception as e:
                self.progress_var.set("")
                messagebox.showerror("错误", f"处理文件时出错: {e}")
        
        # 在后台线程中处理
        thread = threading.Thread(target=process_file_bg)
        thread.daemon = True
        thread.start()

def main():
    """主函数"""
    root = tk.Tk()
    app = MassInterpolatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()