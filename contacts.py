"""Customer contact management for the voice assistant.

This module provides functionality to store, retrieve, and manage customer
contact information in alphabetical order.
"""
import json
import os
from typing import Dict, List, Optional

# Default path for contacts database
CONTACTS_FILE = os.getenv("CONTACTS_FILE", "contacts.json")


def _load_contacts() -> Dict[str, Dict]:
    """Load contacts from JSON file.
    
    Returns:
        Dictionary of contacts with name as key.
    """
    if not os.path.exists(CONTACTS_FILE):
        return {}
    try:
        with open(CONTACTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_contacts(contacts: Dict[str, Dict]) -> None:
    """Save contacts to JSON file.
    
    Args:
        contacts: Dictionary of contacts to save.
    """
    try:
        with open(CONTACTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(contacts, f, ensure_ascii=False, indent=2)
    except IOError as e:
        raise RuntimeError(f"Fehler beim Speichern der Kontakte: {e}")


def add_contact(name: str, phone: Optional[str] = None, email: Optional[str] = None,
                address: Optional[str] = None, notes: Optional[str] = None) -> Dict:
    """Add or update a customer contact.
    
    Args:
        name: Customer name (required, case-insensitive key).
        phone: Phone number.
        email: Email address.
        address: Physical address.
        notes: Additional notes.
    
    Returns:
        Dictionary with success status and message.
    """
    if not name or not name.strip():
        return {"success": False, "message": "Name ist erforderlich"}
    
    contacts = _load_contacts()
    name_key = name.strip().lower()
    
    # Create or update contact
    contact = contacts.get(name_key, {})
    contact["name"] = name.strip()
    if phone is not None:
        contact["phone"] = phone.strip()
    if email is not None:
        contact["email"] = email.strip()
    if address is not None:
        contact["address"] = address.strip()
    if notes is not None:
        contact["notes"] = notes.strip()
    
    contacts[name_key] = contact
    _save_contacts(contacts)
    
    action = "aktualisiert" if name_key in contacts else "hinzugefügt"
    return {
        "success": True,
        "message": f"Kontakt {contact['name']} {action}",
        "contact": contact
    }


def list_contacts(search: Optional[str] = None) -> Dict:
    """List all contacts in alphabetical order.
    
    Args:
        search: Optional search term to filter contacts by name.
    
    Returns:
        Dictionary with list of contacts sorted alphabetically.
    """
    contacts = _load_contacts()
    
    if not contacts:
        return {
            "success": True,
            "contacts": [],
            "count": 0,
            "message": "Keine Kontakte vorhanden"
        }
    
    # Filter by search term if provided
    if search:
        search_lower = search.lower()
        filtered = {k: v for k, v in contacts.items() 
                   if search_lower in k or search_lower in v.get("name", "").lower()}
    else:
        filtered = contacts
    
    # Sort alphabetically by name (case-insensitive)
    sorted_contacts = sorted(filtered.values(), key=lambda x: x.get("name", "").lower())
    
    return {
        "success": True,
        "contacts": sorted_contacts,
        "count": len(sorted_contacts),
        "message": f"{len(sorted_contacts)} Kontakt(e) gefunden"
    }


def get_contact(name: str) -> Dict:
    """Get a specific contact by name.
    
    Args:
        name: Customer name to search for.
    
    Returns:
        Dictionary with contact information or error message.
    """
    if not name or not name.strip():
        return {"success": False, "message": "Name ist erforderlich"}
    
    contacts = _load_contacts()
    name_key = name.strip().lower()
    
    contact = contacts.get(name_key)
    if not contact:
        return {
            "success": False,
            "message": f"Kontakt '{name}' nicht gefunden"
        }
    
    return {
        "success": True,
        "contact": contact,
        "message": f"Kontakt {contact['name']} gefunden"
    }


def delete_contact(name: str) -> Dict:
    """Delete a customer contact.
    
    Args:
        name: Customer name to delete.
    
    Returns:
        Dictionary with success status and message.
    """
    if not name or not name.strip():
        return {"success": False, "message": "Name ist erforderlich"}
    
    contacts = _load_contacts()
    name_key = name.strip().lower()
    
    if name_key not in contacts:
        return {
            "success": False,
            "message": f"Kontakt '{name}' nicht gefunden"
        }
    
    deleted_name = contacts[name_key]["name"]
    del contacts[name_key]
    _save_contacts(contacts)
    
    return {
        "success": True,
        "message": f"Kontakt {deleted_name} gelöscht"
    }
