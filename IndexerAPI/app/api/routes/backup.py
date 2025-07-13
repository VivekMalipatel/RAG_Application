from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from services.vector_store import get_global_vector_store
from services.database_persistence import get_global_db_persistence
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/backup/trigger")
async def trigger_backup():
    try:
        vector_store = get_global_vector_store()
        vector_store.save()
        await vector_store._save_to_s3()
        logger.info("Vector store backup triggered")
        
        db_persistence = get_global_db_persistence()
        await db_persistence.backup_to_s3()
        logger.info("Database backup triggered")
        
        return JSONResponse(
            content={"message": "Backup operations triggered successfully"},
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error triggering backup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")

@router.post("/restore/trigger")
async def trigger_restore():
    try:
        vector_store = get_global_vector_store()
        success = await vector_store._load_from_s3()
        if success:
            logger.info("Vector store restored from S3")
        else:
            logger.warning("No vector store backup found in S3")
        
        db_persistence = get_global_db_persistence()
        success = await db_persistence.restore_from_s3()
        if success:
            logger.info("Database restored from S3")
        else:
            logger.warning("No database backup found in S3")
        
        return JSONResponse(
            content={"message": "Restore operations completed"},
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error triggering restore: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")

@router.get("/backup/status")
async def backup_status():
    try:
        status = {
            "vector_store": {
                "local_exists": False,
                "s3_exists": False
            },
            "database": {
                "local_exists": False,
                "s3_exists": False
            }
        }
        
        import os
        vs = get_global_vector_store()
        idx_path = os.path.join(vs.index_dir, 'faiss.index')
        map_path = os.path.join(vs.index_dir, 'mapping.pkl')
        status["vector_store"]["local_exists"] = os.path.exists(idx_path) and os.path.exists(map_path)
        
        status["vector_store"]["s3_exists"] = await vs.s3_handler.object_exists(f"{vs.s3_prefix}faiss.index") and await vs.s3_handler.object_exists(f"{vs.s3_prefix}mapping.pkl")
        
        db_persist = get_global_db_persistence()
        status["database"]["local_exists"] = os.path.exists(db_persist.db_path)
        status["database"]["s3_exists"] = await db_persist.s3_handler.object_exists(db_persist.s3_key)
        
        return JSONResponse(content=status, status_code=200)
        
    except Exception as e:
        logger.error(f"Error checking backup status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
