import sqlite3
import os
import glob
import re
import time
import pickle
import numpy as np
import torch

def init_db(save_path, db_name="adversarial_attack.db", db_path=None):
    """Initializes the SQLite database with WAL mode for concurrency."""
    if db_path is None:
        db_path = os.path.join(save_path, db_name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    # PC directories
    pc_path = os.path.join(save_path, "orig/")
    adv_path = os.path.join(save_path, "adv/")
    os.makedirs(os.path.dirname(pc_path), exist_ok=True)
    os.makedirs(os.path.dirname(adv_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path, timeout=30)
    
    # 1. Enable WAL Mode (Crucial for multi-GPU performance)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    
    with conn:
        # 2. Tasks Table (The Queue)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attack TEXT NOT NULL,
                model TEXT NOT NULL,
                dataset TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                status TEXT DEFAULT 'PENDING',
                worker_id TEXT,
                UNIQUE(attack, model, dataset, sample_id) -- Prevents duplicate work
            )
        """)
        
        # 3. Results Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attack_results (
                task_id INTEGER,
                sample_id TEXT NOT NULL,
                dataset TEXT NOT NULL,
                orig_output BLOB, -- Serialized dict of boxes/scores
                orig_pc_path TEXT,    -- Path to the .npy file on disk
                adv_output BLOB,
                adv_pc_path TEXT,
                gt_boxes BLOB, -- Serialized dict of boxes
                gt_labels BLOB, -- Serialized dict of labels
                FOREIGN KEY(task_id) REFERENCES tasks(id)
            )
        """)
        
        # 4. Create Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sample ON tasks(sample_id);")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_attack_results_task ON attack_results(task_id);")
        
        
    print(f"Database initialized at {db_path}")
    conn.close()

def check_available(db_path, attack_name, model_name, dataset_name, sample_id, worker_id):
    """
    Attempts to claim a specific sample ID. Continues working on an unfinished sample after crashing.
    Returns True if this worker 'owns' the sample now.
    Returns False if it's already finished or being worked on by someone else.
    """
    with sqlite3.connect(db_path, timeout=60) as conn:
        cursor = conn.cursor()
        # Create entry if it does not exist yet
        cursor.execute("""
            INSERT INTO tasks (dataset, attack, model, sample_id, status, worker_id)
            VALUES (?, ?, ?, ?,'PROCESSING',?)
            ON CONFLICT(dataset, attack, model, sample_id) DO NOTHING;
        """, (dataset_name, attack_name, model_name, str(sample_id), worker_id))
        # Atomic 'Check and Set'
        cursor.execute("""
            UPDATE tasks 
            SET status = 'PROCESSING', worker_id = ?
            WHERE attack = ? AND model = ? AND dataset = ? AND sample_id = ? AND (status = 'PENDING' OR (status = 'PROCESSING' AND worker_id = ?))
            RETURNING id;
        """, (worker_id, attack_name, model_name, dataset_name, str(sample_id), worker_id))
        
        result = cursor.fetchone()
        return result is not None  # If we got an ID back, we claimed it!
    
def check_any_taken_incomplete(db_path, attack_name, model_name, dataset_name, worker_id):
    """
    Attempts to claim a specific sample ID. Continues working on an unfinished sample after crashing.
    Returns True if there are any incomplete samples left
    Returns False if all claimed samples have been completed
    """
    with sqlite3.connect(db_path, timeout=60) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 1
            FROM tasks
            WHERE attack = ? AND model = ? AND dataset = ? AND worker_id = ? AND status = 'PROCESSING'
            LIMIT 1;
        """, (attack_name, model_name, dataset_name, worker_id))
        return cursor.fetchone() is not None

def progress(db_path):
    conn = sqlite3.connect(db_path, timeout=30)

    cursor = conn.execute("""
        SELECT COUNT(*) 
        FROM tasks
        WHERE status = 'COMPLETED'
    """)

    completed_count = cursor.fetchone()[0]
    return completed_count

def save_res(db_path, pc_base_path,result, orig_pc, adv_result, adv_pc, attack, model, dataset, sample_id, gt_boxes, gt_labels, worker_id):

    """
    Finalizes the task in the DB and saves heavy binary data to disk/BLOBs.
    """
    # 1. Prepare File Paths & Save Binary Point Clouds to Disk (.bin for dataset compatability)
    #orig_filename = f"orig/{dataset}_{sample_id}_orig.bin"
    #adv_filename = f"adv/{dataset}_{sample_id}_{attack}_{model}_adv.bin"
    
    # 1. Prepare File Paths & Save Binary Point Clouds to Disk (.pkl for easier evaluation afterwards)
    orig_filename = f"orig/{dataset}_{sample_id}_orig.pkl"
    adv_filename = f"adv/{dataset}_{sample_id}_{attack}_{model}_adv.pkl"

    orig_full_path = os.path.join(pc_base_path, orig_filename)
    adv_full_path = os.path.join(pc_base_path, adv_filename)

    # Atomic write to disk
    #orig_pc.detach().cpu().numpy().astype(np.float32).tofile(orig_full_path)
    #adv_pc.detach().cpu().numpy().astype(np.float32).tofile(adv_full_path)

    pickle.dump(orig_pc, open(orig_full_path, 'wb'))
    pickle.dump(adv_pc, open(adv_full_path, 'wb'))

    conn = sqlite3.connect(db_path, timeout=60)
    with conn:
        # 2. Update the Task Status and get the ID
        # We target the specific task this worker was just processing
        cursor = conn.execute("""
            UPDATE tasks 
            SET status = 'COMPLETED'
            WHERE dataset = ? AND attack = ? AND model = ? AND sample_id = ? AND worker_id = ? AND (status = 'PENDING' OR status = 'PROCESSING')
            RETURNING id;
        """, (dataset, attack, model, str(sample_id), worker_id))
        
        row = cursor.fetchone()
        if row is None:
            print(f"Error: Could not find active task for {sample_id} to mark as completed.")
            return
        
        task_id = row[0]

        # 3. Serialize Dictionaries/Tensors
        # Using pickle for the dicts (labels, bboxes, scores)
        raw_output_blob = pickle.dumps(result)
        adv_output_blob = pickle.dumps(adv_result)
        gt_boxes_blob = pickle.dumps(gt_boxes)
        gt_labels_blob = pickle.dumps(gt_labels)

        # 4. Insert into attack_results
        conn.execute("""
            INSERT INTO attack_results (
                task_id,
                sample_id,
                dataset,
                orig_output,
                orig_pc_path,
                adv_output,
                adv_pc_path,
                gt_boxes,
                gt_labels
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_id,
            str(sample_id),
            dataset,
            raw_output_blob,
            orig_full_path,
            adv_output_blob,
            adv_full_path,
            gt_boxes_blob,
            gt_labels_blob
        ))

    print(f"Sample {sample_id} saved successfully by {worker_id}.")


def transform_to_db(pkl_path, db_path, pc_base_path, attack, model, dataset):
    """
    Transforms a pickle that was generated previously by the pipeline into the db format with a separate folder containing the adversarial point clouds
    """
    init_db(pkl_path, db_path=db_path)

    pc_path = os.path.join(pc_base_path, "orig/")
    adv_path = os.path.join(pc_base_path, "adv/")
    os.makedirs(os.path.dirname(pc_path), exist_ok=True)
    os.makedirs(os.path.dirname(adv_path), exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=30)

    with conn:
        for res in iter_results_multi(pkl_path):
            # Extract fields from old pickle format
            result = res["result"]
            adv_result = res["adv_result"]
            points = res["points"]
            adv_points = res["adv_points"]
            token = res["name"]
            gt_boxes = res["gt_boxes"]
            gt_labels = res["gt_labels"]

            # Metadata for tasks table
            sample_id = token
            status = "COMPLETED"
            worker_id = 0

            # Insert task
            conn.execute("""
                INSERT OR IGNORE INTO tasks (
                    attack,
                    model,
                    dataset,
                    sample_id,
                    status,
                    worker_id
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                attack,
                model,
                dataset,
                sample_id,
                status,
                worker_id
            ))

            # Retrieve task_id
            task_id = conn.execute("""
                SELECT id
                FROM tasks
                WHERE attack = ?
                AND model = ?
                AND dataset = ?
                AND sample_id = ?
            """, (
                attack,
                model,
                dataset,
                sample_id
            )).fetchone()[0]

            # Serialize data
            raw_output = pickle.dumps(result)
            adv_output = pickle.dumps(adv_result)

            gt_boxes_blob = pickle.dumps(gt_boxes)
            gt_labels_blob = pickle.dumps(gt_labels)

            # save adv_points / points to disk and store path
            #orig_dir = os.path.join(pc_path,f"{attack}_{model}_{sample_id}.pt")
            #adv_dir = os.path.join(adv_path,f"{attack}_{model}_{sample_id}.pt")

            #torch.save(points.detach().cpu(), orig_dir)
            #torch.save(adv_points.detach().cpu(), adv_dir)

            orig_dir = os.path.join(pc_path,f"{dataset}_{sample_id}_orig.bin")
            adv_dir = os.path.join(adv_path,f"{dataset}_{sample_id}_{attack}_{model}_adv.bin")

            points_np = points.detach().cpu().numpy().astype(np.float32)
            adv_np = adv_points.detach().cpu().numpy().astype(np.float32)

            points_np.tofile(orig_dir)
            adv_np.tofile(adv_dir)
        
            # Insert original+adversarial results
            conn.execute("""
                INSERT INTO attack_results (
                    task_id,
                    sample_id,
                    dataset,
                    orig_output,
                    orig_pc_path,
                    adv_output,
                    adv_pc_path,
                    gt_boxes,
                    gt_labels
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                str(sample_id),
                dataset,
                raw_output,
                orig_dir,
                adv_output,
                adv_dir,
                gt_boxes_blob,
                gt_labels_blob
            ))

    print("Migration complete.")




# ---- Pickle load functions ----
def iter_results_multi(base_path, mode="auto"):
    root, ext = os.path.splitext(base_path)

    if mode == "single":
        yield from iter_results(base_path)
        return

    if mode == "multi":
        i = 0
        while True:
            path = f"{root}_{i}{ext}"
            if not os.path.exists(path):
                break
            yield from iter_results(path)
            i += 1
        return

    # auto (safe + efficient): glob once
    shards = sorted(
        glob.glob(f"{root}_[0-9]*{ext}"),
        key=lambda p: int(re.search(r"_(\d+)\.pkl$", p).group(1))
    )
    if shards:
        for p in shards:
            yield from iter_results(p)
    else:
        yield from iter_results(base_path)

def iter_results(file_path):
    """
    Generator that yields one sample at a time from mixed pickle formats:
    - Old format: a single list dumped once
    - New format: many single objects appended
    """
    # print(f"[iter_results] opening {file_path}", flush=True)
    with open(file_path, "rb") as f:
        try:
            t0 = time.time()
            first = pickle.load(f)
            # print(f"[iter_results] first object loaded in {time.time()-t0:.1f}s, type={type(first)}", flush = True)

            # Case 1: old format → list of samples
            if isinstance(first, list):
                for item in first:
                    yield item
            else:
                # Case 2: new / mixed format → first object is one sample
                yield first

            # Case 3: appended samples
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

        except EOFError:
            return

# ---- End Pickle load functions ----

import glob

if __name__ == "__main__":
    # ----------- Configuration & Path Setup -----------
    root_visualizations = r"/path/to/Projects/adversarial-attacks/visualizations"

    # Define the migration matrix
    datasets = ["Kitti", "NuScenes", "Waymo"]
    models = ["CenterPoint", "PointPillars", "PillarNeSt", "FocalFormer3D"]

    # Your example listed 6 attacks; all are included here
    attacks = ["iou_attachment", "iou_detachment", "iou_perturbation", "lidattack", "fgsm", "pgd"]

    # Mapping to handle how model names are cased in your directory tree
    # (e.g., CenterPoint variable maps to 'Centerpoint' folder)
    model_folder_mapping = {
        "CenterPoint": "Centerpoint",
        "PointPillars": "Pointpillars",
        "PillarNeSt": "Pillarnest",
        "FocalFormer3D": "Focalformer3d"
    }

    # ----------- Matrix Iteration -----------
    for dataset in datasets:
        print(f"\n############ {dataset} ############")
        
        for model in models:
            # Rule: Skip FocalFormer3D for Kitti dataset
            if dataset == "Kitti" and model == "FocalFormer3D":
                continue
                
            print(f"'''''''' {model} ''''''''")
            model_folder = model_folder_mapping.get(model, model)
            
            for attack in attacks:
                # 1. Apply Dataset Path Rules
                if attack == "iou_perturbation" and model == "CenterPoint" and dataset == "NuScenes":
                    # Exception Path: Flat in root folder, uses lowercase model name
                    pattern = f"run_{model.lower()}_reduced_iou_perturbation_*"
                    
                elif attack == "lidattack":
                    # Uses the reduced dataset directory structure
                    pattern = f"{model_folder}/reduced_{dataset}/{attack}/run_*"
                    
                else:
                    # Full dataset rule for everything else (attachment, detachment, fgsm, pgd)
                    pattern = f"{model_folder}/{dataset}/{attack}/run_*"
                
                # 2. Locate Folders Using Glob Wildcards
                search_pattern = os.path.join(root_visualizations, pattern)
                matching_dirs = glob.glob(search_pattern)
                
                if not matching_dirs:
                    print(f"  [Skipped] No runs found for: {attack}")
                    continue
                
                # 3. Chronological Selection
                # Sorting alphabetically naturally sorts YYYY-MM-DD timestamps.
                # [-1] safely selects the absolute latest run.
                base_path = sorted(matching_dirs)[-1]
                
                # 4. Define Dynamic Database Destination 
                # Crucial: Ensure db_base_path changes per iteration so runs do not overwrite each other!
                # Change this destination root to match where your target DB environment lives.
                db_base_path = f"/path/to/Projects/adversarial-attacks/result_databases/{dataset}/{model}/{attack}"
                os.makedirs(db_base_path, exist_ok=True)
                
                # 5. Resolve Final File Paths
                pkl_path = os.path.join(base_path, "sample_results.pkl")
                db_path = os.path.join(db_base_path, "results.db")
                pc_base_path = os.path.join(db_base_path, "pc_data")
                
                # 6. Execute Migration
                print(f"  -------- {attack} --------")
                print(f"  Processing latest run: {os.path.basename(base_path)}")
                
                transform_to_db(pkl_path, db_path, pc_base_path, attack, model, dataset)
