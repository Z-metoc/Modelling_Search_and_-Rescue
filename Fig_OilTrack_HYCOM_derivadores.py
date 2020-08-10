#!/home/operador/anaconda3/envs/hycomemerg/bin/python
# -*- coding: utf-8 -*-
# -*- coding: iso-8859-1 -*-
#
# OUT2019
# =============================================================>
# IMPORTACAO DE BIBLIOTECAS E DEFINICAO DE DIRETORIOS:

#import datetime, timFig_OilTrack_HYCOM.pye
import datetime
import os, sys, shutil
from  ww3Funcs import alteraStr, alteraDia, horarios

# Diretorio-chave (somente altera-los caso eles mudem):
dirlocal = '/home/operador/AuxDec_HYCOM/corrente_hycom/Fig_HYCOM_manual'
os.chdir(dirlocal)
os.getcwd()

data = horarios('20190613')                                        # MODIFICAR DATAS DE ACORDO COM AS SAIDAS
base=datetime.date(int(data[0]),int(data[1]),int(data[2]))
numdays = 1                                                     
date_list=[base + datetime.timedelta(days=x) for x in range(0, numdays)]
datai=date_list[0]; datai=datai.strftime('%Y%m%d')
dataf=date_list[-1]; dataf=dataf.strftime('%Y%m%d')
#prog=numdays*4
#prog=prog-1
prog = 1
# =============================================================>
                                      # MODIFICAR ÁREA CONFORME LOCALIDADE DOS TRACKS
nome = 'N_NE'    
lat_sul = '-7'
lat_norte = '-5'
lon_oeste = '-35.5'
lon_leste = '-33'
pNome1 = 'SAR '
plon1 = -34.05
plat1 = -6.16
pNome2 = 'Natal '
plon2 = -35.17
plat2 = -5.87
pNome3 = 'Fortaleza '
plon3 = -38.54
plat3 = -3.77
#pNome3 = 'Salvador '
#plon3 = -38.55
#plat3 = -13.01
skip = 8

# =============================================================>
# CONTINUACAO DO PROCESSAMENTO:

from numpy import ma
import numpy as np
import scipy
import matplotlib as mpl
#mpl.use('Agg') # Force matplotlib to not use any Xwindows backend.
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.cm as cm
from netCDF4 import Dataset

# Criando arrays de datas e prog para as figuras
tit_horas=['00','06','12','18'] 
titfig=[]
for ttt in range(0,numdays,1):
   aux=date_list[ttt]
   dt=aux.strftime('%Y%m%d')
   for hh in tit_horas:
      dtit=[]
      dtit=dt+' '+hh+'Z'
      titfig.append(dtit)

titarq=[]
for ttt in range(0,numdays,1):
   aux=date_list[ttt]
   dt=aux.strftime('%Y%m%d')
   for hh in tit_horas:
      dtit=[]
      dtit=dt+'_'+hh
      titarq.append(dtit)


#dirarq='/mnt/nfs/dpns32/data1/operador/previsao/hycom_2_2/proc/ATLo0.04/expt_03.0/data/floats'
dirarq=dirlocal
arq=dirarq+'/floats_out.safe'

os.system("cat "+arq+" | tr -s '[:blank:]' ',' > dados_float_final.txt")

fl = []; fl2 = []
tempo = []; tempo2 = []
nivel = []; nivel2 = []
Lon = []; Lon2 = []
Lat = []; Lat2 = []
prof = []; prof2 = []

import glob
exec("f = open('dados_float_final.txt','r')")
M=""
M = f.readlines()
f.close()
for i in range(0,len(M)):
   fl.append(((M[i]).split(','))[1])
   tempo.append(((M[i]).split(','))[2])
   nivel.append(((M[i]).split(','))[3])
   Lon.append(np.float(((M[i]).split(','))[4])) 
   Lat.append(np.float(((M[i]).split(','))[5]))
   prof.append(np.float(((M[i]).split(','))[6]))

ti=tempo[0]
tf=tempo[-1]

# Filtragem dos dados necessarios:
#A = 441
#for i in range(0,len(fl),A):
#    fl2.append(fl[i])
#    tempo2.append(tempo[i])
#    Lon2.append(Lon[i])
#    Lat2.append(Lat[i])

lat_s=float(lat_sul)
lat_n=float(lat_norte)
lat_media=-((-lat_s)+(-lat_n))/2
lat_media=str(lat_media)

dados='/mnt/nfs/dpns32/data1/operador/previsao/hycom_2_2/hycom2netcdf/Ncdf/HYCOM_1_12_20190801_20191023_sup.nc'
nc_fid=Dataset(dados, 'r')
lat=nc_fid.variables['Latitude'][:]
lon=nc_fid.variables['Longitude'][:]
lons,lats=plab.meshgrid(lon,lat)
lat_s=float(lat_sul)
lat_n=float(lat_norte)
lat_media=-((-lat_s)+(-lat_n))/2
lat_media=str(lat_media)
#u=nc_fid.variables['u_velocity'][:]; u=np.squeeze(u)
#v=nc_fid.variables['v_velocity'][:]; v=np.squeeze(v)
u=nc_fid.variables['u_velocity'][:,:,:]; u=np.squeeze(u)
v=nc_fid.variables['v_velocity'][:,:,:]; v=np.squeeze(v)
u0=np.copy(u)  
v0=np.copy(v) 

for ii in range(0,int(prog)):
   uu=(u0[ii,:,:])
   uu=np.squeeze(uu)
   vv=(v0[ii,:,:])
   vv=np.squeeze(vv)
   for i in range(0,733):
      for j in range(0,601):
         if uu[i,j]>1.0e+30:
            uu[i,j]=0.0
         if vv[i,j]>1.0e+30:
            vv[i,j]=0.0 

#   MM=np.sqrt((uu*uu)+(vv*vv))/1000
   M=1.94*np.sqrt((uu*uu) + (vv*vv))
   aux=np.size(Lon)-1
   

   fig=plt.figure()
   rect = fig.patch
   rect.set_facecolor('white')
   m=Basemap(projection='merc',llcrnrlat=float(lat_sul),urcrnrlat=float(lat_norte),\
   llcrnrlon=float(lon_oeste),urcrnrlon=float(lon_leste),lat_ts=float(lat_media),resolution='h')
   x,y=m(lons,lats)
#   cores=scipy.linspace(0,4.5,num=10)
#   CS=m.contourf(x,y,M,levels=cores,cmap=plt.cm.Blues)
   m.drawcoastlines()
   m.drawlsmask(land_color='green',ocean_color='lightblue',lakes=True)
   m.fillcontinents(color='green',lake_color='lightblue')
   m.drawparallels(np.arange(float(lat_sul),float(lat_norte),1), labels=[1,0,0,0],fmt='%g')
   m.drawmeridians(np.arange(float(lon_oeste),float(lon_leste),1), labels=[0,0,0,1])
   m.drawcountries(linewidth=0.8, color='k', antialiased=1, ax=None, zorder=None)
   m.drawstates(linewidth=0.5, color='k', antialiased=1, ax=None, zorder=None)
#   Q=m.quiver(x[::skip, ::skip], y[::skip, ::skip], uu[::skip, ::skip], vv[::skip, ::skip],\
#   scale=5,units='inches',scale_units='inches',width=0.008,headlength=5,headwidth=3.5,headaxislength=5,\
#   pivot='tail',color='k',minshaft=2,minlength=1)
 
#   for i in range(0,aux,1):
#      if (321 <= int(fl[i]) <= 323 or 345 <= int(fl[i]) <= 345 or 365 <= int(fl[i]) <= 372 or 393 <= int(fl[i]) <= 396):
#         print fl[ii]
#         A,B = m(Lon[i], Lat[i])
#         m.plot(A, B, 'wo-', markersize=1)
#         if (float(tempo[i]) == float(ti)):
#            A,B = m(Lon[i], Lat[i])
#            m.plot(A, B, 'gv-', markersize=2)
#         elif (float(tempo[i]) == float(tf)):
#            A,B = m(Lon[i], Lat[i])
#            m.plot(A, B, 'rs-', markersize=2)
#      elif (316 <= int(fl[i]) <= 319  or 379 <= int(fl[i]) <= 387 or 400 <= int(fl[i]) <= 417):
#         A,B = m(Lon[i], Lat[i])
#         m.plot(A, B, 'y*-', markersize=1)
#         if (float(tempo[i]) == float(ti)):
#            A,B = m(Lon[i], Lat[i]) 
#            m.plot(A, B, 'gv-', markersize=2)
#         elif (float(tempo[i]) == float(tf)):
#            A,B = m(Lon[i], Lat[i])
#            m.plot(A, B, 'rs-', markersize=2)
#      elif (337 <= int(fl[i]) <= 343 or 358 <= int(fl[i]) <= 363 or 390 <= int(fl[i]) <= 391 or 421 <= int(fl[i]) <= 433):
#         A,B = m(Lon[i], Lat[i])
#         m.plot(A, B, 'g+-', markersize=1)
#         if (float(tempo[i]) == float(ti)):
#            A,B = m(Lon[i], Lat[i])
#            m.plot(A, B, 'gv-', markersize=2)
#         elif (float(tempo[i]) == float(tf)):
#            A,B = m(Lon[i], Lat[i])
#            m.plot(A, B, 'rs-', markersize=2)
      
   for i in range(0,aux,1):
      if (10 <= int(fl[i]) <= 15 or 18 <= int(fl[i]) <= 23 or 26 <= int(fl[i]) <= 31 or 34  <= int(fl[i]) <= 36 or 38 <= int(fl[i]) <=39 or 42 <= int(fl[i]) <= 44 or 46 <= int(fl[i]) <= 47  or 50 <= int(fl[i]) <= 55):
#         print fl[ii]
         A,B = m(Lon[i], Lat[i])
         m.plot(A, B, 'w.-', markersize=2)
         if (float(tempo[i]) == float(ti)):
            A,B = m(Lon[i], Lat[i])
            m.plot(A, B, 'wv-', markersize=2)
         elif (float(tempo[i]) == float(tf)):
            A,B = m(Lon[i], Lat[i])
            m.plot(A, B, 'rs-', markersize=2)
      elif (37 == int(fl[i]) ):
         A,B = m(Lon[i], Lat[i])
         m.plot(A, B, 'g.-', markersize=4)
         if (float(tempo[i]) == float(ti)):
            A,B = m(Lon[i], Lat[i]) 
            m.plot(A, B, 'gv-', markersize=4)
         elif (float(tempo[i]) == float(tf)):
            A,B = m(Lon[i], Lat[i])
            m.plot(A, B, 'gs-', markersize=4)
      elif (45 == int(fl[i])):
         A,B = m(Lon[i], Lat[i])
         m.plot(A, B, 'b.-', markersize=4)
         if (float(tempo[i]) == float(ti)):
            A,B = m(Lon[i], Lat[i])
            m.plot(A, B, 'bv-', markersize=4)
         elif (float(tempo[i]) == float(tf)):
            A,B = m(Lon[i], Lat[i])
            m.plot(A, B, 'bs-', markersize=4)
#      elif (379 <= int(fli]) <= 387 or 390 <= int(fl[i]) <= 391 or 393 <= int(fl[i]) <= 396):
#         A,B = m(Lon[i], Lat[i])
#         m.plot(A, B, 'w.-', markersize=2)
#         if (float(tempo[i]) == float(ti)):
#            A,B = m(Lon[i], Lat[i])
#            m.plot(A, B, 'wv-', markersize=2)
#         elif (float(tempo[i]) == float(tf)):
#            A,B = m(Lon[i], Lat[i])
#            m.plot(A, B, 'rs-', markersize=2)
#      elif (400 <= int(fl[i]) <= 417):
#         A,B = m(Lon[i], Lat[i])
#         m.plot(A, B, 'w.-', markersize=2)
#         if (float(tempo[i]) == float(ti)):
#            A,B = m(Lon[i], Lat[i])
#            m.plot(A, B, 'wv-', markersize=2)
#         elif (float(tempo[i]) == float(tf)):
#            A,B = m(Lon[i], Lat[i])
#            m.plot(A, B, 'rs-', markersize=2)
#      elif (421 <= int(fl[i]) <= 433):
#         A,B = m(Lon[i], Lat[i])
#         m.plot(A, B, 'w.-', markersize=2)
#         if (float(tempo[i]) == float(ti)):
#            A,B = m(Lon[i], Lat[i])
#            m.plot(A, B, 'wv-', markersize=2)
#         elif (float(tempo[i]) == float(tf)):
#            A,B = m(Lon[i], Lat[i])
#            m.plot(A, B, 'rs-', markersize=2)
      
#   A,B = m(Lon[ii], Lat[ii])
#   m.plot(A, B, 'go-', markersize=4)

   plt.hold(True)   
   A,B = m(plon1,plat1)
   m.plot(A, B,'ko', markersize=3)
#   plt.text(A,B,pNome1, ha='right', color='k')
   A,B = m(plon2,plat2)
   m.plot(A, B,'ko', markersize=5)
   plt.text(A,B,pNome2, ha='right', color='k')
   A,B = m(plon3,plat3)
   m.plot(A,B,'ko', markersize=5)
   plt.text(A,B,pNome3, ha='right', color='k')
 
   

   aux=titfig[ii]
   print ('')
   print ('Processando figura do Prognostico:',(aux))
   print ('')
   plt.title('HYCOM 4km CHM/REMO') 
   plt.hold(True)
#   cbar=plt.colorbar(CS, format='%.1f')
#   cbar.ax.set_ylabel('corrente [n'u'ós]')
#   plt.savefig('/home/operador/AuxDec_HYCOM/corrente_hycom/Fig_HYCOM_manual/oiltrack/oiltrack_'+titarq[ii]+'Z.png')
   plt.savefig('/home/operador/AuxDec_HYCOM/corrente_hycom/Fig_HYCOM_manual/oiltrack/oiltrack.png')

quit()
