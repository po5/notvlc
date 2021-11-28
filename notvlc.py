from vapoursynth import core

def prop_comp(lst, comp_func, prop):
    return comp_func(range(len(lst)), key=lambda i: lst[i][prop])

def stats(ref, clips, stat_func):
    scores = []
    for clip in clips:
        diff = stat_func(clip, ref)
        scores.append(diff.get_frame(0).props)
    return scores

def select(n, f, ref, stat_func, prop, comp_func, debug=False):
    import vsutil
    clips = list(map(vsutil.frame2clip, f))
    scores = stats(ref, clips, stat_func)
    best = prop_comp(scores, comp_func, prop)
    out = f[best]
    if debug:
        out = vsutil.frame2clip(f[best]).text.Text("\n".join([f"Prop: {prop}", *[f"{i}: {s[prop]}" for i, s in enumerate(scores)], f"Best: {best}"])).get_frame(0)
    return out

def bestframeselect(clips, ref, stat_func, prop, comp_func, debug=False):
    """
    Picks the 'best' clip for any given frame using stat functions.

    clips: list of clips
    ref: reference clip, e.g. core.average.Mean(clips) / core.median.Median(clips)
    stat_func: function that adds frame properties, e.g. core.std.PlaneStats / mvsfunc.PlaneCompare / core.butteraugli.butteraugli
    prop: property added by stat_func to compare, e.g. PlaneStatsDiff / PlaneMAE / PlaneRMSE / PlanePSNR / PlaneCov / PlaneCorr / _Diff
    comp_func: function to decide which clip to pick, e.g. min, max
    comp_func: function to decide which value wins, e.g. min, max
    debug: display values of prop for each clip, and which clip was picked
    """
    from functools import partial
    return core.std.ModifyFrame(clip=clips[0], clips=clips, selector=partial(select, ref=ref, stat_func=stat_func, prop=prop, comp_func=comp_func, debug=debug))

def dvd_titles(path, clip, chapters=True, titles=True, title=None, chapter=None):
    """
    Splits a DVD into separate clips for its corresponding titles/chapters.

    :param path:            Path to DVD directory or ISO
    :param clip:            Clip to cut
    :param chapters:        Split titles into chapters
    :param titles:          Categorize chapters by title
    :param title:           Return a single title with this id, 1-indexed
    :param chapter:         Return a single chapter with this id, 1-indexed
    """

    import dvdread
    tits = []
    pos = 0
    cc = 0
    l = len(clip)

    with dvdread.DVD(path) as d:
        d.Open()

        for t in range(1, d.NumberOfTitles+1):
            tit = d.GetTitle(t)

            duration = int(float(tit.FrameRate)*int(tit.PlaybackTime)/(1000/1001*1000))
            if pos+duration > l:
                continue
            title_v = clip[pos:pos+duration]
            if chapters or chapter is not None:
                chaps = []
                tpos = 0

                for c in range(1, tit.NumberOfChapters+1):
                    chap = tit.GetChapter(c)
                    cc += 1
                    cduration = int(float(tit.FrameRate)*int(chap.Length)/(1000/1001*1000))
                    if chapter is None:
                        chaps.append(title_v[tpos:tpos+cduration])
                    elif (titles and chapter == c and (title is None or title == t)) or (not titles and chapter == cc):
                        return title_v[tpos:tpos+cduration]
                    tpos += cduration + 1

                if title is None:
                    tits.append(chaps)
                elif title == t and chapter is None:
                    return chaps
            else:
                if title is None:
                    tits.append(title_v)
                elif title == t:
                    return title_v
            pos += duration + 1

    if not titles:
        tits = [t for c in tits for t in c]

    return tits

def ReplaceEvery(a, b, cycle=5, offsets=[0, 1, 2, 3, 4]):
    """
    Replace a clip with another at a given interval.

    :param a:               Original clip
    :param b:               Replacement clip
    :param cycle:           Cycle length in frames
    :param offsets:         Frames in cycle to replace (see std.SelectEvery)
    """
    offsets_a = set(list(range(cycle))) - set(offsets)
    offsets_a = [x*2 for x in offsets_a]
    offsets_b = [x*2+1 for x in offsets]
    offsets = sorted(offsets_a + offsets_b)
    return core.std.Interleave([a, b]).std.SelectEvery(cycle=cycle*2, offsets=offsets)

def nice_round(x, base=2):
    return base * round(x / base)

def resizeroo(clip, width, height, scale_func=core.resize.Spline36, scale_x=1, scale_y=1, top=False, bottom=False, **more):
    """
    Resize a clip to cover a canvas, centered if necessary.

    :param clip:            Clip to cut
    :param width:           todo
    :param height:          todo
    :param scale_func:      todo
    :param scale_x:         todo
    :param scale_y:         todo
    :param top:             todo
    :param bottom:          todo
    :param **more:          extra keyword arguments to pass to scale_func
    """
    scaled_width = clip.width * scale_x
    scaled_height = clip.height * scale_y
    ratio = max(width / scaled_width, height / scaled_height)
    clip = scale_func(clip, nice_round(scaled_width * ratio), nice_round(scaled_height * ratio), **more)
    crop_left = (clip.width - width) / 2
    crop_top = (clip.height - height) / 2
    crop_right = crop_left
    crop_bottom = crop_top
    if top:
        crop_bottom += crop_top
        crop_top = 0
    if bottom:
        crop_top += crop_bottom
        crop_bottom = 0
    if crop_left % 2 == 1:
        crop_left += 1
        crop_right -= 1
    if crop_top % 2 == 1:
        crop_top += 1
        crop_bottom -= 1
    if crop_left or crop_right or crop_top or crop_bottom:
        clip = core.std.Crop(clip, crop_left, crop_right, crop_top, crop_bottom)
    return clip


def owoverlay(clip, image, scale_func=core.resize.Spline36, **kwargs):
    """
    Overlay a transparent image on top of a clip.
    Meant for credit recreations.

    :param clip:            Clip
    :param image:           Image to be overlayed
    :param scale_func:      Function to scale image to clip dimensions
    :param **kwargs:        Extra keyword arguments to pass to scale_func
    """
    image, image_mask = core.imwri.Read(image, alpha=True)
    if clip.width != image.width or clip.height != image.height:
        image = resizeroo(image, clip.width, clip.height, scale_func, format=clip.format, matrix_s="709", **kwargs)
        image_mask = resizeroo(image_mask, clip.width, clip.height, scale_func, **kwargs)
    return core.std.MaskedMerge(clip, image, image_mask)

def shifteroo(clip, left=0, top=0):
    """
    Shifts a clip, with wraparound.

    :param clip:            Clip to shift
    :param left:            Shift to the left
    :param top:             Shift from the top
    """
    og_w, og_h = clip.width, clip.height
    if left:
        clip = core.std.StackHorizontal([clip, clip])
    if top:
        clip = core.std.StackVertical([clip, clip])
    clip = clip.resize.Bicubic(src_left=left%og_w, src_top=top%og_h)
    clip = core.std.CropAbs(clip, og_w, og_h)
    return clip

def gen_shifts(clip, n, shift, forward=True, backward=True):
    """
    Return shifts of n neighboring frames.
    Meant for credit blending to reduce flicker.

    :param clip:            Clip to shift
    :param n:               How many neighboring frames to shift
    :param shift:           Float of how many pixels the credits scroll each frame 
    """
    shifts = [clip]
    for cur in range(1, n+1):
        if forward:
            shifts.append(clip[1*cur:].resize.Bicubic(src_top=-shift*cur)+clip[0]*cur)
        if backward:
            shifts.append(clip[0]*cur+clip.resize.Bicubic(src_top=shift*cur)[:-1*cur])
    return shifts

def shift_blend(clip, n, shift, forward=True, backward=True):
    """
    Simple function to test shift values
    """
    return core.average.Mean(gen_shifts(clip, n, shift, forward, backward))

def shift_mask(mask, mask_height, shift, forward=True, backward=True, vertical=True):
    """
    Masks scrolling credits by referencing less busy areas of the frame.

    :param mask:            Masked credits clip constrained to the 'stable' area
    :param mask_height:     Height of the 'stable' area
    :param shift:           Float of how many pixels the credits scroll each frame
    """
    import kagefunc as kgf

    mask = mask.std.ShufflePlanes(planes=0, colorfamily=GRAY)
    return_mask = mask
    shifted = 0
    while shifted < mask.height:
        shifted += mask_height
        if forward:
            shift_frames = int(shifted/shift)
            shifted_mask = mask[0]*shift_frames+mask.resize.Bicubic(src_top=shift_frames*shift)[:-1*shift_frames]
            try:
                original_area = kgf.squaremask(mask, width=mask.width, height=shifted, offset_x=0, offset_y=mask.height-shifted).std.Invert()
            except:
                return return_mask
            shifted_mask = core.std.Expr([original_area, shifted_mask], "x y min")
            return_mask = core.std.Expr([return_mask, shifted_mask], "x y max")
    return return_mask

def FalPosCrop(n, f, clip, clip2, planestatsmin):
    if f.props.PlaneStatsMin > planestatsmin:
        return clip
    else:
        return clip2

def Stab3(clp, ts=7, dxmax=None, dymax=None, UVfix=False, FixFalPos=False, zoom=1, mirror=0, PAR=1.0, Prefilter=None, Luma_Exp=1, Fill3pBorders=True):
    """
    same as Stab2 but with some changes

    clp: input clip
    ts: frames to temporal average for better motion estimation (max. 7)
    dxmax: maximum deviation in pixels
    dymax: x, and y should be the same
    UVfix: Fixes the bug of change of HUE in Depan, not need in depan 1.13.1 and up
    FixFalPos: Fixes borders of 3 or more pixels wide. Use along crop(2,2,-2,-2)...
    zoom: maximum zoom factor (1 disabled)
    mirror: Edge filling. 0 off, 15 everything on
    PAR: PAR of your source
    Prefilter: prefilter
    Luma_Exp: Luma Rebuild
    Fill3pBorders: Fixes borders of 3 or less pixels wide.
    """
    from functools import partial
    import math

    dxmax = int(round(clp.width/180)) if dxmax is None else dxmax
    dymax = dxmax if dymax is None else dymax

    Pref = clp if Prefilter is None else Prefilter
    temp = Pref.focus2.TemporalSoften2(ts, 255, 255, 25, 2) # SC thr to 25 otherwise pans will stutter
    rep = temp.rgvs.Repair(Pref.focus2.TemporalSoften2(1, 255, 255, 25, 2), mode=2)
    inter = core.std.Interleave([rep, Pref])

    """
    # temporal stable auto-contrast (better subpixel detection)
    parta = AutoAdjust(inter, temporal_radius=10, auto_balance=False, auto_gain=True, use_interp=True, avg_safety=0.25, dark_limit=10, bright_limit=10, gamma_limit=1.0, dark_exclude=0.05, bright_exclude=0.05, chroma_process=0, scd_threshold=16, input_tv=False, output_tv=False, high_bitdepth=False, debug_view=False) # closed source?
    partb = inter.resize.Bicubic(range_s="full", range_in_s="limited", dither_type="error_diffusion") if Luma_Exp==1 else inter
    Luma_Expa = parta if Luma_Exp==2 else partb
    """
    Luma_Expa = inter.resize.Bicubic(range_s="full", range_in_s="limited", dither_type="error_diffusion") if Luma_Exp==1 else inter

    mdata = core.mv.DepanEstimate(Luma_Expa, pixaspect=PAR, trust=0, dxmax=dxmax, dymax=dymax, zoommax=zoom)
    stabclip = core.mv.DepanCompensate(inter if Prefilter is None else core.std.Interleave([rep, clp]), data=mdata, offset=-1, mirror=mirror, pixaspect=PAR, matchfields=False, subpixel=2)
    stabclip = stabclip.std.SelectEvery(cycle=2, offsets=0)
    stabcrop = stabclip

    if FixFalPos:
        thick = 3.0 if Fill3pBorders else 2.0  # removing >2px wide borders
        cropx = dxmax*2
        ratiox = math.ceil(99-thick/cropx*100)
        
        crop1 = stabcrop.fb.FillBorders(0,0,0,cropx).std.PlaneStats()
        crop2 = stabcrop.fb.FillBorders(0,cropx,0,0).std.PlaneStats()
        crop3 = stabcrop.fb.FillBorders(0,0,cropx,0).std.PlaneStats()
        crop4 = stabcrop.fb.FillBorders(cropx,0,0,0).std.PlaneStats()
        planestatsmin = clp.std.BlankClip().std.PlaneStats().get_frame(0).props.PlaneStatsMin
        stabcrop = core.std.FrameEval(stabcrop, partial(FalPosCrop, clip=stabcrop, clip2=crop1, planestatsmin=planestatsmin), prop_src=crop1)
        stabcrop = core.std.FrameEval(stabcrop, partial(FalPosCrop, clip=stabcrop, clip2=crop2, planestatsmin=planestatsmin), prop_src=crop2)
        stabcrop = core.std.FrameEval(stabcrop, partial(FalPosCrop, clip=stabcrop, clip2=crop3, planestatsmin=planestatsmin), prop_src=crop3)
        stabcrop = core.std.FrameEval(stabcrop, partial(FalPosCrop, clip=stabcrop, clip2=crop4, planestatsmin=planestatsmin), prop_src=crop4)

    """
    bpad = Fill3pBorders # pad if not even and subsampling requires it

    stabclip.FillBorders(pad=0 if bpad is None else 1, FixFalPos=stabcrop if FixFalPos else None) if Fill3pBorders else stabcrop
    """

    """
    if UVfix:
        lumaf=stabcrop
        

        blue=round((AverageChromaU(clp) - AverageChromaU()) * 256.0)
        red=round((AverageChromaV(clp) - AverageChromaV()) * 256.0)
         
        scr = Dither_convert_8_to_16(clp)
        scr = clp.SmoothTweak16(saturation=1.0,hue1=min(384,blue),hue2=min(384,red),HQ=true)
        scr = clp.DitherPost(stacked=true,prot=false,mode=6,y=1,slice=false)

        stabcrop = Mergeluma(lumaf, scr)
    """

    return stabcrop

# mostly pasted from lvsfunc.deinterlace.TIVTC_VFR
def magicivtc(clip, tfm_in="tfm_in.txt", tdec_in="tdec_in.txt"):
    import lvsfunc as lvf

    tfm_in = Path(tfm_in).resolve()
    tdec_in = Path(tdec_in).resolve()

    # TIVTC can't write files into directories that don't exist
    if not tfm_in.parent.exists():
        tfm_in.parent.mkdir(parents=True)
    if not tdec_in.parent.exists():
        tdec_in.parent.mkdir(parents=True)

    if not (tfm_in.exists() and tdec_in.exists()):
        ivtc_clip = core.tivtc.TFM(clip, output=str(tfm_in)).tivtc.TDecimate(mode=1, hint=True, output=str(tdec_in))

        with lvf.render.get_render_progress() as pr:
            task = pr.add_task("Analyzing frames...", total=ivtc_clip.num_frames)

            def _cb(n, total):
                pr.update(task, advance=1)

            with open(os.devnull, "wb") as dn:
                ivtc_clip.output(dn, progress_update=_cb)

        # Allow it to properly finish writing the logs
        time.sleep(0.5)
        del ivtc_clip # Releases the clip, and in turn the filter (prevents it from erroring out)

    return clip.tivtc.TFM(input=str(tfm_in)).tivtc.TDecimate(mode=1, hint=True, input=str(tdec_in), tfmIn=str(tfm_in))

def magicdecimate(fm, tdec_in="tdec_in.txt"):
    import lvsfunc as lvf

    tdec_in = Path(tdec_in).resolve()

    # TIVTC can't write files into directories that don't exist
    if not tdec_in.parent.exists():
        tdec_in.parent.mkdir(parents=True)

    if not tdec_in.exists():
        ivtc_clip = fm.tivtc.TDecimate(mode=1, hint=True, output=str(tdec_in))

        with lvf.render.get_render_progress() as pr:
            task = pr.add_task("Analyzing frames...", total=ivtc_clip.num_frames)

            def _cb(n, total):
                pr.update(task, advance=1)

            with open(os.devnull, "wb") as dn:
                ivtc_clip.output(dn, progress_update=_cb)

        # Allow it to properly finish writing the logs
        time.sleep(0.5)
        del ivtc_clip # Releases the clip, and in turn the filter (prevents it from erroring out)

    return fm.tivtc.TDecimate(mode=1, hint=True, input=str(tdec_in))
