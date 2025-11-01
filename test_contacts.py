"""Tests for the contacts module.

Run with: python test_contacts.py
"""
import os
import sys
import json
import tempfile

# Import the module to test
import contacts


def test_add_and_list_contacts():
    """Test adding contacts and listing them in alphabetical order."""
    print("Test: add_and_list_contacts...")
    
    # Add some contacts
    result = contacts.add_contact("Müller Hans", phone="0123-456789", email="hans@example.com")
    assert result["success"], f"Failed to add contact: {result}"
    
    result = contacts.add_contact("Schmidt Anna", phone="0987-654321")
    assert result["success"], f"Failed to add contact: {result}"
    
    result = contacts.add_contact("Becker Thomas", email="thomas@example.com", notes="Stammkunde")
    assert result["success"], f"Failed to add contact: {result}"
    
    # List all contacts - should be in alphabetical order
    result = contacts.list_contacts()
    assert result["success"], "Failed to list contacts"
    assert result["count"] == 3, f"Expected 3 contacts, got {result['count']}"
    
    # Check alphabetical order
    names = [c["name"] for c in result["contacts"]]
    assert names == ["Becker Thomas", "Müller Hans", "Schmidt Anna"], f"Wrong order: {names}"
    
    print("✓ add_and_list_contacts passed")


def test_search_contacts():
    """Test searching for contacts."""
    print("Test: search_contacts...")
    
    # Search for specific contact
    result = contacts.list_contacts(search="Müller")
    assert result["success"], "Failed to search contacts"
    assert result["count"] == 1, f"Expected 1 contact, got {result['count']}"
    assert result["contacts"][0]["name"] == "Müller Hans"
    
    # Search with partial match
    result = contacts.list_contacts(search="Anna")
    assert result["success"], "Failed to search contacts"
    assert result["count"] == 1, f"Expected 1 contact, got {result['count']}"
    
    print("✓ search_contacts passed")


def test_get_contact():
    """Test getting a specific contact."""
    print("Test: get_contact...")
    
    result = contacts.get_contact("Müller Hans")
    assert result["success"], "Failed to get contact"
    assert result["contact"]["phone"] == "0123-456789"
    assert result["contact"]["email"] == "hans@example.com"
    
    # Test with different case
    result = contacts.get_contact("müller hans")
    assert result["success"], "Failed to get contact (case insensitive)"
    
    # Test non-existent contact
    result = contacts.get_contact("Nicht Vorhanden")
    assert not result["success"], "Should not find non-existent contact"
    
    print("✓ get_contact passed")


def test_update_contact():
    """Test updating an existing contact."""
    print("Test: update_contact...")
    
    # Update phone number
    result = contacts.add_contact("Müller Hans", phone="0123-999999")
    assert result["success"], "Failed to update contact"
    
    # Verify update
    result = contacts.get_contact("Müller Hans")
    assert result["success"], "Failed to get updated contact"
    assert result["contact"]["phone"] == "0123-999999", "Phone not updated"
    assert result["contact"]["email"] == "hans@example.com", "Email should remain"
    
    print("✓ update_contact passed")


def test_delete_contact():
    """Test deleting a contact."""
    print("Test: delete_contact...")
    
    result = contacts.delete_contact("Schmidt Anna")
    assert result["success"], "Failed to delete contact"
    
    # Verify deletion
    result = contacts.list_contacts()
    assert result["count"] == 2, f"Expected 2 contacts after deletion, got {result['count']}"
    
    # Try to delete non-existent
    result = contacts.delete_contact("Schmidt Anna")
    assert not result["success"], "Should not delete non-existent contact"
    
    print("✓ delete_contact passed")


def test_empty_name():
    """Test validation for empty names."""
    print("Test: empty_name...")
    
    result = contacts.add_contact("")
    assert not result["success"], "Should not allow empty name"
    
    result = contacts.get_contact("")
    assert not result["success"], "Should not get with empty name"
    
    print("✓ empty_name passed")


def main():
    """Run all tests with temporary contacts file."""
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    # Override the contacts file path
    original_file = contacts.CONTACTS_FILE
    contacts.CONTACTS_FILE = temp_file
    
    try:
        print("\n" + "="*60)
        print("Running contacts module tests")
        print("="*60 + "\n")
        
        test_add_and_list_contacts()
        test_search_contacts()
        test_get_contact()
        test_update_contact()
        test_delete_contact()
        test_empty_name()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60 + "\n")
        
    finally:
        # Restore original file path and cleanup
        contacts.CONTACTS_FILE = original_file
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    main()
