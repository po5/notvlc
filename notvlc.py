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
