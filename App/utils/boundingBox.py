def union_bbox(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

def contains_ratio(inner, outer):
    """
    Qué fracción del área de 'inner' está cubierta por 'outer'.
    Si inner está totalmente dentro de outer => ~1.0
    """
    xA = max(inner[0], outer[0])
    yA = max(inner[1], outer[1])
    xB = min(inner[2], outer[2])
    yB = min(inner[3], outer[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_inner = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    return (inter / area_inner) if area_inner > 0 else 0.0

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    den = areaA + areaB - inter
    return float(inter) / float(den) if den > 0 else 0.0

def merge_overlapping_bboxes(bboxes, iou_thr=0.15, contain_thr=0.90):
    """
    Fusiona bboxes que:
      - tienen IoU >= iou_thr, o
      - una contiene a la otra (contain_ratio >= contain_thr) en algún sentido.
    Devuelve lista de bboxes "clusterizadas".
    """
    if not bboxes:
        return []

    merged = [list(b) for b in bboxes]

    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue

            cur = tuple(merged[i])
            used[i] = True

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue

                oth = tuple(merged[j])

                ov = iou(cur, oth)
                cont = max(contains_ratio(cur, oth), contains_ratio(oth, cur))

                if ov >= iou_thr or cont >= contain_thr:
                    cur = union_bbox(cur, oth)
                    used[j] = True
                    changed = True

            out.append(list(cur))

        merged = out

    return [tuple(b) for b in merged]

def merge_tracks(tracks, iou_thr=0.20, contain_thr=0.90):
    """
    Une tracks cuyo bbox se solapa/contiene mucho.
    Conserva el track con id menor y actualiza su bbox a la unión.
    """
    if not tracks:
        return []

    tracks = [dict(t) for t in tracks]  # copia superficial
    used = [False] * len(tracks)
    out = []

    for i in range(len(tracks)):
        if used[i]:
            continue

        base = tracks[i]
        used[i] = True
        cur_bbox = base["bbox"]
        cur_id = base["id"]
        cur_last = base["last_seen"]

        changed = True
        while changed:
            changed = False
            for j in range(i + 1, len(tracks)):
                if used[j]:
                    continue
                oth = tracks[j]
                ov = iou(cur_bbox, oth["bbox"])
                cont = max(contains_ratio(cur_bbox, oth["bbox"]), contains_ratio(oth["bbox"], cur_bbox))

                if ov >= iou_thr or cont >= contain_thr:
                    # merge: me quedo con el ID menor
                    cur_bbox = union_bbox(cur_bbox, oth["bbox"])
                    cur_id = min(cur_id, oth["id"])
                    cur_last = max(cur_last, oth["last_seen"])
                    used[j] = True
                    changed = True

        out.append({"id": cur_id, "bbox": cur_bbox, "last_seen": cur_last})

    # opcional: ordenar por id para que quede prolijo
    out.sort(key=lambda t: t["id"])
    return out


