import numpy as np
import numba
import matplotlib.pyplot as plt

###物理定数####################################
c =2.998e8 #光速[m/s]
epsilon_0 = 8.854e-12 #真空の誘電率[F/m]
mu_0 = 4.0e-7*np.pi #真空の透磁率
sigma = 0.0 #真空の電気電動率
z_0 = 376.7343091821 #インピーダンス
##############################################


###DEFINE(model)##############################
VACUUM = 1 #真空領域
METAL = 2 #金属板
##############################################

def get_dy_by_courant_condition(dx:float) -> float:
    """クーラン条件を満たす時間ステップを求める

    Args:
        dx (float): メッシュ幅[m]

    Returns:
        float: クーラン条件をギリギリ満たす時間[s]
    """    

    dt = dx / (2**0.5 * c)

    return dt


def run_tm_2d(mx:int, my:int, dx:float, dy:float, nstep:int, dt:float, freq:float
    ) -> (np.ndarray, np.ndarray, np.ndarray):
    
    """[summary]
    Args:
        mx (int) :X方向のメッシュ数
        my (int) :Y方向のメッシュ数
        dx (float) : X方向のメッシュの長さ[m]
        dy (float) : Y方向のメッシュの長さ[m]
        nstep (int) : 総ステップ数
        dt (float) : 1ステップあたりの時間[s]
        freq (float) : 波源の周波数[Hz]

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): 時間ステップごとのZ方向電場・X方向磁場・Y方向磁場
    """
    
    # return用の配列
    list_ele_z = []
    list_mag_x = []
    list_mag_y = []

    # 初期化
    ele_z, mag_x, mag_y = _init_field(mx, my)

    # 物体のモデリング
    region_info = _modeling(mx, my)

    # mainループ
    fstep = 0.0
    while fstep < float(nstep):

        print(f"********** step : {fstep}   start ********** ")
        # 励振（平面波）
        ele_z = _plane_wave_source(
            ele_z, dt, fstep, freq
        )

        #電場
        ele_z = _update_electronic_field(
            ele_z, mag_x, mag_y, region_info, dx, dy, dt, fstep, freq
        )
        
        fstep += 0.5

        #磁場

        mag_x, mag_y = _update_magnetic_field(
            ele_z, mag_x, mag_y, dx, dy, dt, region_info
        )


        fstep += 0.5

        list_ele_z.append(ele_z)
        list_mag_x.append(mag_x)
        list_mag_y.append(mag_y)

    return np.array(list_ele_z), np.array(list_mag_x), np.array(list_mag_y)

def _init_field(mx:int, my:int) -> (np.ndarray, np.ndarray, np.ndarray):
    """電場と磁場の初期化（TM法なので、電場はZ方向のみ、磁場はx,y方向のみ）

    Args:
        mx (int): X方向のメッシュ数
        my (int): Y方向のメッシュ数

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): z方向電場、x方向磁場、y方向磁場
    """    
    #電場の初期化
    electron_field_z = np.zeros((my + 1, mx + 1))

    #磁場の初期化
    magnetic_field_x = np.zeros((my + 1, mx + 1))
    magnetic_field_y = np.zeros((my + 1, mx + 1))
    
    return electron_field_z, magnetic_field_x, magnetic_field_y

def _modeling(mx:int, my:int) -> np.ndarray:
    """物体の情報を作成する

    Args:
        mx (int): X方向のメッシュ数
        my (int): y方向のメッシュ数

    Returns:
        np.ndarray: 物体の情報を格納した配列（mx*my）

    Note:
        初期設定：X方向の中間位置にY方向の半分の長さの金属板を置く
    """    

    region_info = np.full((my, mx), VACUUM)

    # 中間地点に金属板を置く
    for iy in range(0, int(my/2)):
        region_info[iy, int(mx/2)] = METAL

    return region_info.astype("int")

def _update_electronic_field(
    ele_z:np.ndarray, mag_x:np.ndarray, mag_y:np.ndarray, 
    region_info:np.ndarray, dx:float, dy:float, dt:float, fstep:float, freq:float
) -> np.ndarray:
    """1ループ分の計算をして電場を更新

    Args:
        ele_z (np.ndarray): [description]
        mag_x (np.ndarray): [description]
        mag_y (np.ndarray): [description]
        region_info (np.ndarray) : 
        dx (float) : x方向の1メッシュのサイズ
        dy (float) : y方向の1メッシュのサイズ
        dt (float) : 1ステップの長さ（時間）
        fstep (float) : 現在のステップ数(1/2も含む)
        freq (float) : 波源の周波数

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): Z方向電場
    """    


    # 電場の更新
    ele_z = _calc_inner_electoronic_field(
        ele_z, mag_x, 
        mag_y, dx, dy, dt, region_info, freq
    )

    return ele_z

@numba.jit(nopython=True)
def _calc_abc(new_ele_z:np.ndarray, ele_z:np.ndarray, dt:float, dx:float, dy:float, freq:float)->np.ndarray:
    """吸収境界条件の計算（１次のMur条件）

    Args:
        new_ele_z (np.ndarray): Z方向の電場(内部計算の更新後)
        ele_z (np.ndarray): Z方向の電場(内部計算の更新前）
        dt (float): 時間ステップ
        dx (float): １メッシュの大きさ（X方向）
        dy (float): １メッシュの大きさ（Y方向）
        freq (float) : 波源の周波数

    Returns:
        np.ndarray: 境界条件が考慮された電場（Z方向）
    """  

    ####x####
    murx = (c*dt - dx) / (c*dt + dx)

    # 左端
    ix = 0
    for iy in range(0, ele_z.shape[0]):
        new_ele_z[iy, ix] = ele_z[iy, ix + 1] + murx*(new_ele_z[iy, ix + 1] - ele_z[iy, ix])
    
    # 右端
    ix = ele_z.shape[1] - 1
    for iy in range(0, ele_z.shape[0]):
        new_ele_z[iy, ix] = ele_z[iy, ix - 1] + murx*(new_ele_z[iy, ix - 1] - ele_z[iy, ix])


    #####y#####
    mury = (c*dt - dy)/ (c*dt + dy)

    #　上端
    iy = 0
    for ix in range(0, ele_z.shape[1]):
        new_ele_z[iy, ix] = ele_z[iy + 1, ix] + mury*(new_ele_z[iy + 1, ix] - ele_z[iy, ix])
    
    # 下端
    iy = ele_z.shape[0] - 1
    for ix in range(0, ele_z.shape[1]):
        new_ele_z[iy, ix] = ele_z[iy - 1, ix] + mury*(new_ele_z[iy - 1, ix] - ele_z[iy, ix])

    return new_ele_z
    
@numba.jit(nopython=True)
def _calc_inner_electoronic_field(
    ele_z:np.ndarray, mag_x:np.ndarray, mag_y:np.ndarray,
    dx:float, dy:float, dt:float, region_info:np.ndarray, freq:float
    ) -> np.ndarray:
    """内部の電場を更新する（Maxwell）

    Args:
        ele_z (np.ndarray): Z方向の電場
        mag_x (np.ndarray): X方向の磁場
        mag_y (np.ndarray): Y方向の磁場
        dx (float) : x方向の1メッシュのサイズ
        dy (float) : y方向の1メッシュのサイズ
        dt (float) : 1ステップの長さ（時間）
        region_info (np.ndarray) : モデル領域の情報
        freq (float) : 波源の周波数

    Return:
        (np.ndarray) : Z方向の電場（更新後）
    """

    cez = (1-sigma*dt/(2*epsilon_0))/(1+sigma*dt/(2*epsilon_0))
    cezlx = (dt/epsilon_0)/(1+sigma*dt/(2*epsilon_0))/dx
    cezly = (dt/epsilon_0)/(1+sigma*dt/(2*epsilon_0))/dy

    new_ele_z = np.copy(ele_z)

    # 0 ~ my + 1のうち、1 ~ myのメッシュ（内部）を更新する
    for iy in range(1, new_ele_z.shape[0] - 1):
        # 0 ~ mx + 1のうち、1 ~ mxのメッシュ（内部）を更新する
        for ix in range(1, new_ele_z.shape[1] - 1):
            #真空領域
            if (region_info[iy, ix] == VACUUM):
                new_ele_z[iy, ix] = cez*ele_z[iy, ix]\
                                    + cezlx*(mag_y[iy, ix] - mag_y[iy, ix - 1])\
                                    - cezly*(mag_x[iy, ix] - mag_x[iy - 1, ix])
            elif (region_info[iy, ix] == METAL):
                new_ele_z[iy, ix] = 0.0

    #境界吸収条件
    new_ele_z = _calc_abc(new_ele_z, ele_z, dt, dx, dy, freq)

    return new_ele_z

@numba.jit(nopython=True)
def _plane_wave_source(
    ele_z:np.ndarray, dt:float, fstep:float, freq:float
    ) -> np.ndarray:
    '''平面波を与える
    
    Args:
        ele_z (np.ndarray) : Z方向の電場
        dt (float) : 時間ステップ幅
        fstep (float) : 現在のステップ数
        freq (float) : 波源の周波数

    Return:
        (np.ndarray, np.ndarray) : Z方向の電場

    '''    

    
    t = dt*fstep
    
    ix = 1
    for iy in range(1, ele_z.shape[0]-1):
        ele_z[iy, ix] = np.cos(2*np.pi*freq*t)
        #mag_y[iy, ix] = ele_z[iy, ix]/z_0
    return ele_z

def _update_magnetic_field(
    ele_z:np.ndarray, mag_x:np.ndarray, mag_y:np.ndarray,
    dx:float, dy:float, dt:float, region_info:np.ndarray
    ) -> (np.ndarray, np.ndarray):
    """1ループ分の計算をして電場を更新

    Args:
        ele_z (np.ndarray): Z方向の電場（更新前）
        mag_x (np.ndarray): X方向の磁場
        mag_y (np.ndarray): Y方向の磁場
        dx (float) : x方向の1メッシュのサイズ
        dy (float) : y方向の1メッシュのサイズ
        dt (float) : 1ステップの長さ（時間）
        region_info (np.ndarray) : モデル領域の情報

    Returns:
        (np.ndarray, np.ndarray): X方向の磁場、Y方向の磁場
    """ 

    # X方向の磁場の更新
    mag_x = _calc_inner_magnetic_field_x(ele_z, mag_x, dy, region_info)

    # Y方向の磁場の更新
    mag_y = _calc_inner_magnetic_field_y(ele_z, mag_y, dx, region_info)
    
    return mag_x, mag_y

@numba.jit(nopython=True)
def _calc_inner_magnetic_field_x(
    ele_z:np.ndarray, mag_x:np.ndarray, dy:float, region_info:np.ndarray)->np.ndarray:
    """内部の磁場の更新（X方向）

    Args:
        ele_z (np.ndarray): Z方向の電場
        mag_x (np.ndarray): X方向の磁場（更新前）
        dy (float): y方向の１メッシュのサイズ
        region_info (np.nadarray): モデル領域の情報

    Returns:
        np.ndarray: X方向の磁場（更新後）
    """  

    chxly = dt/(mu_0*dy)
    new_mag_x = np.copy(mag_x)
    
    chxly = dt/(mu_0*dy)
    for iy in range(0, new_mag_x.shape[0] - 1):
        for ix in range(1, new_mag_x.shape[1] - 1):
            if (region_info[iy, ix] == VACUUM):
                new_mag_x[iy, ix] = mag_x[iy, ix] - chxly*(ele_z[iy + 1, ix] - ele_z[iy, ix])
            elif (region_info[iy, ix] == METAL):
                new_mag_x[iy, ix] = 0.0

    return new_mag_x

@numba.jit(nopython=True)
def _calc_inner_magnetic_field_y(
    ele_z:np.ndarray, mag_y:np.ndarray, dx:float, region_info:np.ndarray)->np.ndarray:
    """内部の磁場の更新（Y方向）

    Args:
        ele_z (np.ndarray): Z方向の電場
        mag_y (np.ndarray): Y方向の磁場（更新前）
        dx (float): x方向の１メッシュのサイズ
        region_info (np.ndarray): モデル領域の情報

    Returns:
        np.ndarray: Y方向の磁場（更新後）
    """    

    chylx = dt/(mu_0*dx)
    new_mag_y = np.copy(mag_y)

    for iy in range(1, new_mag_y.shape[0] - 1):
        for ix in range(0, new_mag_y.shape[1] - 1):
            if (region_info[iy, ix] == VACUUM):
                new_mag_y[iy, ix] = mag_y[iy, ix] + chylx*(ele_z[iy, ix + 1] - ele_z[iy, ix])
            elif (region_info[iy, ix] == METAL):
                new_mag_y[iy, ix] = 0.0
            
    return new_mag_y


if __name__ == "__main__":

    freq = 5e9 #5GHz
    lambda_0 = c / freq #波長[m]

    N_lambda_0 = 50 #波長の何分の１をメッシュにとるか？

    dx = lambda_0 / N_lambda_0
    dy = dx

    #dt = get_dy_by_courant_condition(dx)
    N_period = 100 # 周期の何分の１を１ステップとするか？
    dt = 1.0/freq/N_period

    ele_z, mag_x, mag_y = run_tm_2d(
        mx = 200, my = 200, dx = dx, dy = dy,
        nstep = 500, dt = dt, freq = freq
    )
    for step, (ele_z_2d, mag_x_2d, mag_y_2d) in enumerate(zip(ele_z, mag_x, mag_y)):
        if (step > 100 )& (step % 5 == 0):
            plt.imshow(ele_z_2d, vmin = -1, vmax = 1)
            plt.colorbar()
            plt.savefig("./src/fdtd/data/ez_" + str(step).zfill(4) + ".png")
            plt.close()

            # plt.imshow(mag_x_2d)
            # plt.colorbar()
            # plt.savefig("./src/fdtd/data/hx_" + str(step).zfill(4) + ".png")
            # plt.close()

            # plt.imshow(mag_y_2d)
            # plt.colorbar()
            # plt.savefig("./src/fdtd/data/hy_" + str(step).zfill(4) + ".png")
            # plt.close()

    pass


