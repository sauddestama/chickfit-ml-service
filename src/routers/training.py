import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks

from ..models.schemas import RetrainingRequest, RetrainingResponse
from ..services.training_service import trigger_model_retraining
from ..models.cnn_model import cnn_model

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active retraining tasks (in production, use Redis or database)
active_tasks = {}

@router.post("/retrain", response_model=RetrainingResponse)
async def retrain_model(
    request: RetrainingRequest, 
    background_tasks: BackgroundTasks
):
    """Trigger model retraining with new data"""
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Add retraining task to background
        background_tasks.add_task(
            run_retraining_task,
            task_id,
            request.admin_id,
            request.notes or ""
        )
        
        # Store task info
        active_tasks[task_id] = {
            'status': 'started',
            'admin_id': request.admin_id,
            'notes': request.notes,
            'started_at': datetime.now(),
            'progress': 0
        }
        
        logger.info(f"Retraining task {task_id} started by admin {request.admin_id}")
        
        return RetrainingResponse(
            success=True,
            message="Model retraining started successfully",
            task_id=task_id
        )
        
    except Exception as e:
        logger.error(f"Error starting retraining: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start model retraining"
        )

@router.get("/retrain/status/{task_id}")
async def get_retraining_status(task_id: str):
    """Get status of a retraining task"""
    try:
        if task_id not in active_tasks:
            raise HTTPException(
                status_code=404,
                detail="Task not found"
            )
        
        task_info = active_tasks[task_id]
        
        return {
            "success": True,
            "task_id": task_id,
            "status": task_info['status'],
            "progress": task_info['progress'],
            "started_at": task_info['started_at'],
            "completed_at": task_info.get('completed_at'),
            "error": task_info.get('error')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving task status"
        )

async def run_retraining_task(task_id: str, admin_id: int, notes: str):
    """Background task for model retraining"""
    try:
        logger.info(f"Starting retraining task {task_id}")
        
        # Update task status
        active_tasks[task_id]['status'] = 'downloading_data'
        active_tasks[task_id]['progress'] = 10
        
        # Trigger the actual retraining process
        result = await trigger_model_retraining(
            admin_id=admin_id,
            notes=notes,
            progress_callback=lambda progress: update_task_progress(task_id, progress)
        )
        
        if result['success']:
            # Update task as completed
            active_tasks[task_id]['status'] = 'completed'
            active_tasks[task_id]['progress'] = 100
            active_tasks[task_id]['completed_at'] = datetime.now()
            active_tasks[task_id]['model_accuracy'] = result.get('accuracy', 0.0)
            
            logger.info(f"Retraining task {task_id} completed successfully")
        else:
            # Update task as failed
            active_tasks[task_id]['status'] = 'failed'
            active_tasks[task_id]['error'] = result.get('error', 'Unknown error')
            active_tasks[task_id]['completed_at'] = datetime.now()
            
            logger.error(f"Retraining task {task_id} failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error in retraining task {task_id}: {e}")
        active_tasks[task_id]['status'] = 'failed'
        active_tasks[task_id]['error'] = str(e)
        active_tasks[task_id]['completed_at'] = datetime.now()

def update_task_progress(task_id: str, progress: int):
    """Update progress of a retraining task"""
    if task_id in active_tasks:
        active_tasks[task_id]['progress'] = progress

@router.get("/retrain/tasks")
async def list_retraining_tasks():
    """List all retraining tasks"""
    try:
        tasks = []
        for task_id, task_info in active_tasks.items():
            tasks.append({
                "task_id": task_id,
                "status": task_info['status'],
                "progress": task_info['progress'],
                "admin_id": task_info['admin_id'],
                "started_at": task_info['started_at'],
                "completed_at": task_info.get('completed_at'),
                "notes": task_info.get('notes')
            })
        
        return {
            "success": True,
            "tasks": tasks
        }
        
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving task list"
        )