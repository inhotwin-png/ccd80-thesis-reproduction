"""Draw maps from the released display grid and native 62 m candidate cells."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from lunar.common import OXIDES

ROOT=Path(__file__).resolve().parent
LABELS=["SiO₂","Al₂O₃","FeO","MgO","CaO","TiO₂"]

def limits(a, trim=False):
    v=a[np.isfinite(a)]
    return np.percentile(v,[2,98]) if trim else (v.min(),v.max())

def region_xy(cut, centre):
    lon, lat=centre
    dx=(cut["longitude"]-lon+180)%360-180
    return (np.deg2rad(dx)*1737.4*np.cos(np.deg2rad(lat)),
            np.deg2rad(cut["latitude"]-lat)*1737.4)

def area_panel(fig, ax, x, y, values, label, trim=False):
    ax.set_facecolor("white")
    ax.add_patch(Circle((0,0),2,facecolor=".86",edgecolor=".35",lw=.6,zorder=-1))
    good=np.isfinite(values)
    low,high=limits(values,trim)
    cols, xi = np.unique(x, return_inverse=True)
    rows, yi = np.unique(y, return_inverse=True)
    dx, dy = np.median(np.diff(cols)), np.median(np.diff(rows))
    native=np.full((len(rows), len(cols)), np.nan)
    native[yi[good], xi[good]]=values[good]
    footprint=(rows[:,None]**2+cols[None,:]**2<=2**2) | np.isfinite(native)
    cmap=plt.get_cmap("RdYlBu_r").with_extremes(bad=".86")
    marker=ax.imshow(native,origin="lower",interpolation="nearest",cmap=cmap,vmin=low,vmax=high,
        extent=(cols[0]-dx/2,cols[-1]+dx/2,rows[0]-dy/2,rows[-1]+dy/2),alpha=footprint.astype(float))
    ax.add_patch(Circle((0,0),2,fill=False,edgecolor=".3",lw=.7,ls="--"))
    ax.set(xlim=(-2.1,2.1),ylim=(-2.1,2.1),aspect="equal",xlabel="East (km)",ylabel="North (km)")
    fig.colorbar(marker,ax=ax,shrink=.72,label=label)

def draw(out):
    with (out/"appendix_C.csv").open() as f: candidates=list(csv.DictReader(f))
    with (out/"heldout_windows.csv").open() as f: validation=list(csv.DictReader(f))
    regions=json.loads((ROOT/"data/representative_regions.json").read_text())
    with np.load(ROOT/"data/global_display.npz") as grid:
        maps=grid["means"].copy()
        maps[grid["valid_count"]==0]=np.nan
    for overlays,name in ((False,"figure2_maps.png"),(True,"figure5_candidates.png")):
        fig,axes=plt.subplots(3,2,figsize=(12,9),layout="constrained")
        for j,(oxide,ax) in enumerate(zip(OXIDES,axes.flat)):
            g=maps[:,:,j]
            low,high=limits(g,True)
            im=ax.imshow(g,vmin=low,vmax=high,extent=(-180,180,-90,90),origin="upper",aspect="auto",cmap=plt.get_cmap("RdYlBu_r").with_extremes(bad=".84"),interpolation="nearest")
            ax.set(title=LABELS[j],xlabel="Longitude (°E)",ylabel="Latitude (°N)")
            fig.colorbar(im,ax=ax,shrink=.7,label="Predicted abundance (wt%)",extend="both")
            if overlays:
                rows=[v for v in candidates if v["oxide"]==oxide]
                ax.scatter([float(v["longitude_e"]) for v in rows],[float(v["latitude_n"]) for v in rows],s=15,facecolors="none",edgecolors="black",linewidths=.65)
                ax.scatter(float(rows[0]["longitude_e"]),float(rows[0]["latitude_n"]),s=85,marker="*",c="gold",edgecolors="black")
        fig.savefig(out/name,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(8,10),layout="constrained")
    for j,(oxide,ax) in enumerate(zip(OXIDES,axes.flat)):
        rows=[r for r in validation if r["oxide"]==oxide]
        for site in sorted({r["site"] for r in rows}):
            part=[r for r in rows if r["site"]==site]
            ax.scatter([float(r["reference_wt_pct"]) for r in part],[float(r["ccd80_prediction"]) for r in part],s=16,label=site)
        bounds=np.array([(float(r["reference_wt_pct"]),float(r["ccd80_prediction"])) for r in rows])
        line=[bounds.min(),bounds.max()]
        ax.plot(line,line,"k--",lw=.7);ax.set(title=LABELS[j],xlabel="Reference (wt%)",ylabel="Held-out prediction (wt%)")
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc="outside lower center",ncol=4)
    fig.savefig(out/"figure4_validation.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(9,12),layout="constrained")
    for j,(oxide,ax) in enumerate(zip(OXIDES,axes.flat)):
        top=next(v for v in candidates if v["oxide"]==oxide)
        with np.load(ROOT/regions[oxide]["file"]) as cut:
            x,y=region_xy(cut,(float(top["longitude_e"]),float(top["latitude_n"])))
            values=np.where(cut["display_valid"][:,j],cut["prediction"][:,j],np.nan)
            area_panel(fig,ax,x,y,values,"Predicted abundance (wt%)")
        lat,lon=float(top["latitude_n"]),float(top["longitude_e"])
        ax.set_title(f"{LABELS[j]}   {abs(lat):.3f}°{'N' if lat>=0 else 'S'}, {abs(lon):.3f}°{'E' if lon>=0 else 'W'}")
    fig.savefig(out/"figure6_regions.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    with np.load(ROOT/"data/input_panels.npz") as src:
        for ax,g,title,unit in zip(axes[0],(src["class_ca_si_4x4_mean"],maps[:,:,4]),("CLASS Ca Kα/Si","Predicted CaO"),("Line intensity ratio","Abundance (wt%)")):
            lo,hi=limits(g,True)
            im=ax.imshow(g,origin="upper",extent=(-180,180,-90,90),aspect="auto",vmin=lo,vmax=hi,cmap="RdYlBu_r")
            ax.set(title=title,xlabel="Longitude (°E)",ylabel="Latitude (°N)");fig.colorbar(im,ax=ax,shrink=.72,label=unit)
        with np.load(ROOT/regions["TiO2"]["file"]) as cut:
            top=next(v for v in candidates if v["oxide"]=="TiO2")
            x,y=region_xy(cut,(float(top["longitude_e"]),float(top["latitude_n"])))
            pred=np.where(cut["display_valid"][:,5],cut["prediction"][:,5],np.nan)
            for ax,val,title,unit in zip(axes[1],(src["mi_mv2_reflectance"],pred),("MI MV2 (749 nm)","Predicted TiO₂"),("Reflectance","Abundance (wt%)")):
                area_panel(fig,ax,x,y,val,unit,True);ax.set_title(title)
                lat,lon=float(top["latitude_n"]),float(top["longitude_e"])
                step=360/2048
                meridians=np.arange(-180,180,step)
                dx=1737.4*np.cos(np.deg2rad(lat))*np.deg2rad(meridians-lon)
                dx=dx[np.abs(dx)<2]
                ax.scatter(dx,np.sqrt(4-dx**2),marker="v",s=18,c="black",zorder=4)
                parallels=np.arange(-90,90,step)
                dy=1737.4*np.deg2rad(parallels-lat)
                dy=dy[np.abs(dy)<2]
                ax.scatter(np.sqrt(4-dy**2),dy,marker="<",s=18,c="black",zorder=4)
    fig.savefig(out/"figure7_input_patterns.png",dpi=150);plt.close(fig)
