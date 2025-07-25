import os
from django.core.management.base import BaseCommand
from django.conf import settings
from rag_app.rag_system import MultilingualRAGSystem


class Command(BaseCommand):
    help = 'Initialize the RAG system with PDF documents'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force-rebuild',
            action='store_true',
            dest='force_rebuild',
            help='Force rebuild of vector store even if it exists',
        )
        parser.add_argument(
            '--file',
            type=str,
            help='Specify the document file to process (supports .pdf, .txt, .docx)',
        )

    def handle(self, *args, **options):
        self.stdout.write('Initializing RAG system...')
        
        try:
            # Initialize RAG system
            rag_system = MultilingualRAGSystem()
            
            # Determine which file to use
            if options.get('file'):
                file_path = os.path.join(settings.PDF_STORAGE_PATH, options['file'])
            else:
                # Try to find any supported document in the data directory
                data_dir = settings.PDF_STORAGE_PATH
                supported_extensions = ['.pdf', '.txt', '.docx', '.doc']
                
                file_path = None
                for filename in os.listdir(data_dir):
                    if any(filename.lower().endswith(ext) for ext in supported_extensions):
                        file_path = os.path.join(data_dir, filename)
                        break
                
                if not file_path:
                    self.stdout.write(
                        self.style.ERROR(f'No supported document found in {data_dir}. Supported formats: {", ".join(supported_extensions)}')
                    )
                    return
            
            if not os.path.exists(file_path):
                self.stdout.write(
                    self.style.ERROR(f'Document file not found at {file_path}')
                )
                return
            
            force_rebuild = options.get('force_rebuild', False)
            
            self.stdout.write(f'Processing document: {file_path}')
            if force_rebuild:
                self.stdout.write('Force rebuilding vector store...')
            
            rag_system.initialize_system(file_path, force_rebuild=force_rebuild)
            
            # Get system stats
            stats = rag_system.get_system_stats()
            
            self.stdout.write(
                self.style.SUCCESS('RAG system initialized successfully!')
            )
            self.stdout.write(f'Total documents: {stats["total_documents"]}')
            
            if 'document_languages' in stats:
                self.stdout.write('Document languages:')
                for lang, count in stats['document_languages'].items():
                    self.stdout.write(f'  {lang}: {count} chunks')
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to initialize RAG system: {str(e)}')
            )
