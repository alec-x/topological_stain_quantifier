# https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application
import numpy as np
from PIL import ImageTk, Image
Image.MAX_IMAGE_PIXELS = None
import tkinter as tk
from tkinter import filedialog
import cv2
import ntpath
import scipy.ndimage as ndi
from skimage.measure import block_reduce

XPAD = 8
YPAD = 4
IMAGE_SIZE = 96

def normalize255(img):
    return img/np.max(img)*255

def normalize_arb(img, max):
    res = ((img/max)*255)
    res[res > 255] = 255
    return res.astype(np.uint8)

def subtract_calc(arr, num_erodes, sigma=10):
    print('\nCalculating middle subtraction')
    orig_size = arr.shape
    subtraction = cv2.resize(arr, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
    subtraction = ndi.gaussian_filter(subtraction, sigma)
    subtraction = subtraction/np.max(subtraction)*100
    subtraction[subtraction < 50] = 0
    subtraction[subtraction >= 50] = 1
    subtraction = ndi.binary_dilation(subtraction.astype(np.uint8), iterations=num_erodes)
    subtraction = 1 - subtraction
    subtraction = cv2.resize(subtraction.astype(float), dsize=orig_size, interpolation=cv2.INTER_CUBIC)
    print('\n Middle subtraction calculations complete')
    return subtraction

def adaptive_threshold(arr, dia):
    print("\nCalculating adaptive thresholding")
    new_arr = np.zeros_like(arr)

    sample_area = dia
    s_len = int(sample_area/2)
    
    for i in range(s_len,arr.shape[0], sample_area):
        for j in range(s_len,arr.shape[1], sample_area):
            curr_area = arr[i-s_len:i+s_len, j-s_len:j+s_len]
            background = np.percentile(curr_area, 0.5)
            curr_area = curr_area - background
            curr_area[curr_area < 0] = 0
            new_arr[i-s_len:i+s_len, j-s_len:j+s_len] = curr_area
        print(f"Processing {i/arr.shape[0]:.2%}", end='\r')
    print("\nAdapative thresholding complete")
    return new_arr

def threshold(arr, low, high):
    print("\nCalculating final thresholds")
    filtered_arr = normalize255(np.array(arr))
    filtered_arr[filtered_arr < low] = 0
    filtered_arr[filtered_arr > high] = 0
    filtered_arr[filtered_arr > 0] = 1
    print("\nFinal thresholds complete")
    return filtered_arr

def sum_px(arr, num_divs):
    print("Calculating NET Score in target blocks")
    targ_div = (int(arr.shape[0]/num_divs), int(arr.shape[1]/num_divs))
    print("NET Score calculations complete")
    return block_reduce(arr, targ_div, np.sum)

class PathBar(tk.Frame):
    def __browse_file(self, textbox):
        path = filedialog.askopenfilename(parent=self, title="Please select NET EGFP image")
        if len(path) > 0:
            textbox.delete(0, "end")
            textbox.insert(0,path)
            print("Selected path: " + path)
            return 0
        return 1

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
       
        in_egfp_lbl = tk.Label(self, text="EGFP path:")
        in_egfp_txt = tk.Entry(self, textvariable=parent.in_egfp)
        in_dapi_lbl = tk.Label(self, text="DAPI path:")
        in_dapi_txt = tk.Entry(self, textvariable=parent.in_dapi)

        in_egfp_lbl.pack(side="left", padx=XPAD, pady=YPAD)
        in_egfp_txt.pack(side="left", fill="x", expand=True, padx=XPAD, pady=YPAD)
        in_dapi_lbl.pack(side="left", padx=XPAD, pady=YPAD)
        in_dapi_txt.pack(side="left", fill="x", expand=True, padx=XPAD, pady=YPAD)

        in_egfp_txt.bind("<ButtonPress-1>", lambda event: self.__browse_file(in_egfp_txt))
        in_dapi_txt.bind("<ButtonPress-1>", lambda event: self.__browse_file(in_dapi_txt))

class SaveBar(tk.Frame):
    def __browse_folder(self, textbox):
        path = filedialog.askdirectory(parent=self, title="Please select output directory")
        if len(path) > 0:
            textbox.delete(0, "end")
            textbox.insert(0,path)
            print("Selected directory: " + path)
            return 0
        else:
    
            return 1

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
       
        self.out_path_lbl = tk.Label(self, text="Save dir:      ")
        self.out_path_txt = tk.Entry(self, textvariable=parent.out_path)
        self.slash_lbl = tk.Label(self, text="/")
        self.out_name_txt = tk.Entry(self, textvariable=parent.out_name)
        self.png_txt = tk.Label(self, text=".png")
        self.save_btn = tk.Button(self, text="Save", width=20)
        
        self.out_path_lbl.grid(column=0, row=0, sticky="w", padx=XPAD, pady=YPAD)
        self.out_path_txt.grid(column=1, row=0, sticky="ew", padx=XPAD, pady=YPAD)
        self.slash_lbl.grid(column=2, row=0, sticky="e", padx=XPAD, pady=YPAD)
        self.out_name_txt.grid(column=3, row=0, sticky="ew", padx=XPAD, pady=YPAD)
        self.png_txt.grid(column=4, row=0, sticky="w", pady=YPAD)
        self.save_btn.grid(column=5, row=0, sticky="ew", padx=XPAD, pady=YPAD)
        
        for i, w in enumerate([0,3,0,1,0,0]):
            self.columnconfigure(i, weight=w)
        self.out_path_txt.bind("<ButtonPress-1>", lambda event: self.__browse_folder(self.out_path_txt))

class SimpleSlider(tk.Frame):
    def __init__(self, parent, txt, lims, res, sld_var, default, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.sld_var = sld_var
        self.sld_lbl = tk.Label(self, text=txt)
        self.sld = tk.Scale(self, 
                    from_=lims[0],to=lims[1], resolution=res, 
                    orient=tk.HORIZONTAL, 
                    showvalue=0)
        self.sld_txt = tk.Label(self, textvariable=sld_var)
        self.sld.bind("<ButtonRelease-1>", self.updateValue)
        
        self.sld_lbl.grid(column=0, row=0, sticky="w")
        self.sld.grid(column=1, row=0, sticky="ew")
        self.sld_txt.grid(column=2, row=0, sticky="w")        
        self.sld_var.set(default)
        self.sld.set(default)
        for i, w in enumerate([0,1,0]):
            self.columnconfigure(i, weight=w)
    
    def updateValue(self, event):
        self.sld_var.set(self.sld.get())

class SlidersBar(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.grid_line_sld = SimpleSlider(self, "Num Grid Lines", (10,400),5,parent.num_divs,60)
        self.mid_erode_sld = SimpleSlider(self, "Middle Erosion", (0, 100), 1, parent.mid_erode, 20)
        self.adapt_sld = SimpleSlider(self, "Adaptive Background Removal Dia.", (50, 1000), 5, parent.adapt_dia, 100)
        self.low_sld = SimpleSlider(self, "Px Value Lim (low)", (0, 254), 1, parent.low, 6)
        self.high_sld = SimpleSlider(self, "Px Value Lim (high)", (1, 255), 1, parent.high, 30)
        self.max_sld = SimpleSlider(self, "Scale (highest)", (100, 20000), 100, parent.max, 1000)

        self.grid_line_sld.grid(column=0, row=0, sticky="ew", columnspan=2)
        self.mid_erode_sld.grid(column=0, row=1, sticky="ew", columnspan=2)
        self.adapt_sld.grid(column=0, row=2, sticky="ew", columnspan=2)
        self.low_sld.grid(column=0, row=3, sticky="ew")
        self.low_sld.configure(relief="raised", bd=2)
        self.high_sld.grid(column=1, row=3, sticky="ew")
        self.high_sld.configure(relief="raised", bd=2)
        self.max_sld.grid(column=0, row=4, sticky="ew", columnspan=2)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

class Display(tk.Frame):
    def motion(self, event):
        self.x, self.y = event.x, event.y

    def render_image(self, img_arr):
        try:
            if img_arr == None:
                return
        except ValueError as e:
            pass
   
        can = self.can
        disp_width = can.winfo_width()
        disp_height = can.winfo_height()

        img =  Image.fromarray(img_arr)
        img = img.resize((disp_width, disp_height), resample=Image.NEAREST)
        can.delete("all")
        can.image = ImageTk.PhotoImage(img)
        can.create_image((0,0), image=can.image, anchor="nw")

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.x = 0
        self.y = 0
        self.parent = parent
        self.can = tk.Canvas(self, bg='white')
        self.can.image = None
        self.can.pack(expand=True, fill="both")

        self.can.bind('<Motion>', lambda event: self.motion(event))

class ScoreBar(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.score_lbl = tk.Label(self, text="NET Score:")
        self.score_num_lbl = tk.Label(self, textvariable=parent.curr_score, width=20)
        self.update_btn = tk.Button(self, text="Calculate", width=20)

        self.score_lbl.pack(side="left")
        self.score_num_lbl.pack(side="left")
        self.score_num_lbl.configure(relief="groove", bd=2, background="#28c5e0")
        self.update_btn.pack(side="right")

class MainApplication(tk.Frame):

    def switch_zoom(self):
        try:
            if not self.block_arr:
                return
        except ValueError:
            pass

        x_max, y_max = self.areadisplay.can.winfo_width(), self.areadisplay.can.winfo_height()
        x_spc = x_max/self.num_divs.get()/2
        y_spc = y_max/self.num_divs.get()/2
        x, y = self.areadisplay.x, self.areadisplay.y
       
        self.areadisplay.render_image(normalize_arb(self.block_arr, self.max.get()))
        self.areadisplay.can.create_rectangle(x-x_spc, y-y_spc, x+x_spc, y+y_spc,outline="orange")
        
        x_big = int(self.areadisplay.x / x_max * self.arr.shape[0])
        y_big = int(self.areadisplay.y / y_max * self.arr.shape[1])
        x_spc = int(x_spc * self.filter_arr.shape[0] / x_max)
        y_spc = int(y_spc * self.filter_arr.shape[1] / y_max)

        self.zoomdisplay.render_image(self.filter_arr[y_big-y_spc:y_big+y_spc, x_big-x_spc:x_big+x_spc])
        self.update_net_score()

    def save_NET_map(self, img: np.array, name: str):
        base_name = ntpath.basename(self.in_egfp.get())
        cv2.imwrite(f"{base_name}/{name}.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def update_all(self):
        # This is nested to attempt to make it all a bit faster . . .
        self.filter_arr = adaptive_threshold(
            np.multiply(self.orig_arr, subtract_calc(self.dapi_arr, self.mid_erode.get())), 
            self.adapt_dia.get())

        self.arr = threshold(self.filter_arr, self.low.get(), self.high.get())
        self.max.set(np.max(sum_px(self.arr, self.num_divs.get())))
        self.update_grid()

    def update_grid(self):
        self.block_arr = sum_px(self.arr, self.num_divs.get())
        self.areadisplay.render_image(normalize_arb(self.block_arr, self.max.get()))

    def load_image(self, path, type: str):
        try:
            if type == "egfp":
                self.arr = np.array(Image.open(path))
                
                len_x, len_y = self.arr.shape
                trim_x = int(len_x/20)
                trim_y = int(len_y/20)
                self.arr = self.arr[trim_x:len_x-trim_x, trim_y:len_y-trim_y]
                self.orig_arr = np.array(self.arr)  
                if not self.dapi_arr:
                    self.areadisplay.render_image(self.arr)
                
            else: 
                self.dapi_arr = np.array(Image.open(path))
                len_x, len_y = self.dapi_arr.shape
                trim_x = int(len_x/20)
                trim_y = int(len_y/20)
                self.dapi_arr = self.dapi_arr[trim_x:len_x-trim_x, trim_y:len_y-trim_y]
                
                self.update_all()
        except AttributeError:
            pass

    def update_net_score(self):
        x, y = self.areadisplay.x, self.areadisplay.y
        x_dis, y_dis = self.areadisplay.can.winfo_width(), self.areadisplay.can.winfo_height()
        x_max, y_max = self.block_arr.shape
        x_ind, y_ind = int(x / x_dis * x_max), int(y / y_dis * y_max)
        self.curr_score.set(f'{self.block_arr[y_ind, x_ind]: .3f}')


    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.in_egfp = tk.StringVar(self)
        self.in_dapi = tk.StringVar(self)
        self.out_path = tk.StringVar(self)
        self.out_name = tk.StringVar(self)

        self.num_divs = tk.IntVar(self)
        self.mid_erode = tk.IntVar(self)
        self.adapt_dia = tk.IntVar(self)
        self.low = tk.IntVar(self)
        self.high = tk.IntVar(self)
        self.max = tk.IntVar(self)
        self.curr_score = tk.StringVar(self)
        
        self.out_name.set("NET_scores")

        self.orig_arr = None
        self.dapi_arr = None
        self.arr = None
        self.block_arr = None
        self.filter_arr = None

        self.parent = parent

        self.pathbar = PathBar(self)
        self.savebar = SaveBar(self)
        self.varsliders = SlidersBar(self)
        self.areadisplay = Display(self)
        self.zoomdisplay = Display(self)
        self.scorebar = ScoreBar(self)

        self.pathbar.pack(side="top", fill="x")
        self.savebar.pack(side="top", fill="x") 
        self.varsliders.pack(side="top", fill="x")
        self.scorebar.pack(side="top", fill="x")
        self.areadisplay.pack(side="left", expand=True, fill="both")
        self.zoomdisplay.pack(side="right", expand=True, fill="both")

        self.in_egfp.trace_add("write", lambda n, i, d: self.load_image(self.in_egfp.get(), "egfp"))
        self.in_dapi.trace_add("write", lambda n, i, d: self.load_image(self.in_dapi.get(), "dapi"))
        self.scorebar.update_btn.bind("<ButtonPress-1>", lambda event: self.update_all())
        self.num_divs.trace_add("write", lambda n, i, d: self.update_grid())
        self.max.trace_add("write", lambda n, i, d: self.update_grid())
        self.areadisplay.can.bind("<ButtonPress-1>", lambda event: self.switch_zoom())

drag_id = None
window_width, window_height = 0, 0

if __name__ == "__main__":
    root = tk.Tk()
    root.title("NET Score Browser")
    root.minsize(700, 500)
    main_app = MainApplication(root)
    main_app.pack(side="top", fill="both", expand=True)

    def stop_drag():
        global drag_id
        try:
            main_app.areadisplay.render_image(main_app.arr)
        except TypeError:
            pass

        try:
            main_app.zoomdisplay.render_image(main_app.arr[500:1000, 500:1000])
        except TypeError:
            pass

        # reset drag_id to be able to detect the start of next dragging
        drag_id = None 

    def dragging(event):
        global drag_id
        global window_height, window_width
        # do nothing if the event is triggered by one of root's children or window size same
        
        cond = (window_width != event.width) and (window_height != event.height)

        if cond:  
            window_width, window_height = event.width,event.height
            # cancel scheduled call to stop_drag
            if drag_id:
                root.after_cancel(drag_id)
            
            # schedule stop_drag
            drag_id = root.after(1000, stop_drag)    


    root.bind('<Configure>', dragging)
    root.mainloop()