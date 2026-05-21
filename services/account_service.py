from firebase_admin import auth

from services.firebase_service import get_firestore_client, initialize_firebase, user_document


def _delete_document_tree(document_ref) -> int:
    deleted = 0
    for collection_ref in document_ref.collections():
        while True:
            snapshots = list(collection_ref.limit(100).stream())
            if not snapshots:
                break
            for snapshot in snapshots:
                deleted += _delete_document_tree(snapshot.reference)

    document_ref.delete()
    return deleted + 1


def delete_account(uid: str) -> dict:
    client = get_firestore_client()
    user_ref = user_document(uid)

    try:
        deleted_documents = int(client.recursive_delete(user_ref) or 0)
    except AttributeError:
        deleted_documents = _delete_document_tree(user_ref)

    initialize_firebase()
    auth.delete_user(uid)

    return {
        "deleted_firestore_documents": deleted_documents,
        "deleted_firebase_user": True,
    }
