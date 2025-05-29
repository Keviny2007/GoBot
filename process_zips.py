from dlgo.data.parallel_processor import GoDataProcessor

if __name__ == '__main__':
    processor = GoDataProcessor()
    generator = processor.load_go_data('test', 100, use_generator=False)