#!/usr/bin/env python3
"""
Test metadata-passage mapping functionality
"""

from metadata_db_v2 import MetadataDBV2
import json

def test_metadata_with_passages():
    """Test searching metadata with passages"""
    
    print("="*80)
    print("Test: Metadata with Passage Mapping")
    print("="*80)
    
    with MetadataDBV2() as db:
        # Test 1: Search for Leonardo DiCaprio
        print("\n[Test 1] Search: 'Leonardo DiCaprio'")
        print("-" * 80)
        
        results = db.get_metadata_with_passages("Leonardo DiCaprio")
        
        print(f"Found {len(results)} results\n")
        
        for idx, result in enumerate(results, 1):
            print(f"\n{'='*80}")
            print(f"Result {idx}: {result['title']}")
            print(f"Type: {result['type']} - {result['subtype']}")
            print(f"\n[Matched Paths]")
            for path in result['matched_paths'][:3]:
                print(f"  - {path}")
            
            print(f"\n[Associated Passages: {len(result['passages'])}]")
            for p_idx, passage in enumerate(result['passages'], 1):
                print(f"\n  Passage {p_idx}:")
                print(f"    QA ID: {passage['passage_id']}")
                print(f"    Title: {passage['title']}")
                print(f"    Sentences: {passage['sentence_count']}")
                print(f"    Content: {passage['content'][:300]}...")
        
        # Test 2: Search for Argentina
        print("\n\n" + "="*80)
        print("[Test 2] Search: 'Argentina'")
        print("-" * 80)
        
        results = db.get_metadata_with_passages("Argentina")
        
        print(f"Found {len(results)} results\n")
        
        for idx, result in enumerate(results[:2], 1):  # Show first 2
            print(f"\n{'='*80}")
            print(f"Result {idx}: {result['title']}")
            print(f"Type: {result['type']} - {result['subtype']}")
            
            print(f"\n[Associated Passages: {len(result['passages'])}]")
            for p_idx, passage in enumerate(result['passages'][:1], 1):  # Show first passage
                print(f"\n  Passage {p_idx}:")
                print(f"    QA ID: {passage['passage_id']}")
                print(f"    Sentences: {passage['sentence_count']}")
                print(f"    Content: {passage['content'][:200]}...")
        
        # Test 3: Get specific metadata with passages
        print("\n\n" + "="*80)
        print("[Test 3] Direct Lookup: 'The Wolf of Wall Street (2013 film)'")
        print("-" * 80)
        
        passages = db.get_passages_for_metadata("The Wolf of Wall Street (2013 film)")
        
        print(f"Found {len(passages)} passage(s)\n")
        
        for idx, passage in enumerate(passages, 1):
            print(f"\nPassage {idx}:")
            print(f"  QA ID: {passage['passage_id']}")
            print(f"  Title: {passage['title']}")
            print(f"  Sentences: {passage['sentence_count']}")
            print(f"\n  Full Content:")
            print(f"  {passage['content'][:500]}...")
    
    print("\n" + "="*80)
    print("✓ Test complete!")
    print("="*80)


if __name__ == "__main__":
    test_metadata_with_passages()
