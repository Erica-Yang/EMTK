import os
from .meta_voc import register_meta_voc
from .meta_coco import register_meta_coco
from .builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog


# -------- COCO -------- #
def register_all_coco(root="datasets"):

    METASPLITS = [
        # ("coco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        # ("coco14_trainval_base", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        # ("coco14_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        # ("coco14_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        # ("coco14_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("removecoco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k_10shot.json"),
        # ("removecoco14_trainval_all", "coco/trainval2014", "cocosplit/seed0/merged_5shot_base_rmNew.json"),
        # ("removecoco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k_10shot_max1000pCls.json"),
        # ("trainval_all_1shot_seed0", "coal/train", "coal/fewshot_train_shot1_seed3.json"),
        # ("coal_test_all", "coal/valid", "coal/instances_valCoal_all_new.json"),
        ("coBa_coal_test_all", "coco/trainval2014", "cocosplit/datasplit/5k_cocoBaseCoalTest.json"),
        ("coBa_coal_test_all_less", "coco/trainval2014", "cocosplit/datasplit/5k_cocoBaseCoalTest_less.json"),
        # ("coal_test_less", "coal/valid", "coal/instances_valCoal_all_new_less.json"),
    ]
    for prefix in ["all", "novel"]: #all
        for shot in [1, 2, 3, 5, 10, 30]: #1
            for seed in range(3): #0
                name = "coco14_trainval_{}_{}shot_seed{}".format(prefix, shot, seed) # coco14_trainval_all_1_seed0
                METASPLITS.append((name, "coco/trainval2014", ""))

                if prefix == "all":
                    name = "removecoco14_trainval_{}_{}shot_seed{}".format(prefix, shot, seed) #removecoco14_trainval_all_1shot_seed0
                    METASPLITS.append((name, "coco/trainval2014", ""))

    for name, imgdir, annofile in METASPLITS: #'coco14_trainval_all'
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# -------- PASCAL VOC -------- #
def register_all_voc(root="datasets"):

    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
    ]
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(30):
                        seed = "_seed{}".format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )

                        if prefix == "all":
                            name = "removevoc_{}_trainval_{}{}_{}shot{}".format(
                                year, prefix, sid, shot, seed
                            )
                            METASPLITS.append(
                                (name, dirname, img_file, keepclasses, sid)
                            )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_voc(
            name,
            _get_builtin_metadata("voc_fewshot"),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


register_all_coco()
register_all_voc()