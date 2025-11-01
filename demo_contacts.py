"""Integration test demonstrating contact management functionality.

This test shows how the contacts feature works and can be used.
Does not require external API keys.
"""
import os
import tempfile
import json

# Set up temp file for testing
test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
test_file.close()
os.environ['CONTACTS_FILE'] = test_file.name

from contacts import add_contact, list_contacts, get_contact, delete_contact


def demonstrate_contact_management():
    """Demonstrate the contact management features."""
    
    print("=" * 70)
    print("Kundenkontakte-Verwaltung - Demonstrationstest")
    print("=" * 70)
    print()
    
    # 1. Add multiple contacts
    print("1. Hinzufügen von Kontakten:")
    print("-" * 70)
    
    contacts_to_add = [
        ("Zimmermann Peter", "0221-123456", "peter.zimmermann@email.de", "Stammkunde seit 2020"),
        ("Becker Maria", "0172-9876543", "maria.becker@web.de", "Badezimmer Renovierung"),
        ("Müller Thomas", "0151-234567", "t.mueller@gmail.com", None),
        ("Schmidt Anna", "0211-555123", "anna.schmidt@online.de", "Küche"),
        ("Albrecht Klaus", "0177-888999", None, "Elektroarbeiten")
    ]
    
    for name, phone, email, notes in contacts_to_add:
        result = add_contact(name, phone=phone, email=email, notes=notes)
        print(f"✓ {name:20} -> {result['message']}")
    
    print()
    
    # 2. List all contacts (alphabetically)
    print("2. Alle Kontakte (alphabetisch sortiert):")
    print("-" * 70)
    
    result = list_contacts()
    print(f"Anzahl Kontakte: {result['count']}")
    print()
    
    for i, contact in enumerate(result['contacts'], 1):
        print(f"{i}. {contact['name']}")
        if 'phone' in contact:
            print(f"   Tel:   {contact['phone']}")
        if 'email' in contact:
            print(f"   Email: {contact['email']}")
        if 'notes' in contact:
            print(f"   Info:  {contact['notes']}")
        print()
    
    # 3. Search for contacts
    print("3. Suche nach 'Müller':")
    print("-" * 70)
    
    result = list_contacts(search="Müller")
    for contact in result['contacts']:
        print(f"Gefunden: {contact['name']} - {contact.get('phone', 'keine Tel.')}")
    print()
    
    # 4. Get specific contact
    print("4. Details für 'Schmidt Anna' abrufen:")
    print("-" * 70)
    
    result = get_contact("Schmidt Anna")
    if result['success']:
        contact = result['contact']
        print(json.dumps(contact, ensure_ascii=False, indent=2))
    print()
    
    # 5. Update contact
    print("5. Telefonnummer von 'Müller Thomas' aktualisieren:")
    print("-" * 70)
    
    result = add_contact("Müller Thomas", phone="0151-999888")
    print(f"✓ {result['message']}")
    
    result = get_contact("Müller Thomas")
    print(f"Neue Telefonnummer: {result['contact']['phone']}")
    print()
    
    # 6. Delete contact
    print("6. Kontakt 'Albrecht Klaus' löschen:")
    print("-" * 70)
    
    result = delete_contact("Albrecht Klaus")
    print(f"✓ {result['message']}")
    
    result = list_contacts()
    print(f"Verbleibende Kontakte: {result['count']}")
    print()
    
    # 7. Show alphabetical order is maintained
    print("7. Alphabetische Sortierung (Namen):")
    print("-" * 70)
    
    result = list_contacts()
    names = [c['name'] for c in result['contacts']]
    for name in names:
        print(f"  • {name}")
    print()
    
    # Verify alphabetical order
    is_sorted = names == sorted(names)
    print(f"✓ Liste ist alphabetisch sortiert: {is_sorted}")
    print()
    
    print("=" * 70)
    print("Test erfolgreich abgeschlossen!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        demonstrate_contact_management()
    finally:
        # Clean up
        if os.path.exists(test_file.name):
            os.remove(test_file.name)
