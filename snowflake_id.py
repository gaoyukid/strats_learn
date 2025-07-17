import time
import threading
from typing import Tuple

class BigIntSnowflakeGenerator:
    def __init__(self, worker_id: int = 0, epoch: int = 1609459200000):  # 2021-01-01
        """
        支持大整数的 Snowflake ID 生成器
        
        参数:
        worker_id: 工作节点ID (0-1023)
        epoch: 自定义起始时间戳 (毫秒)
        """
        # 验证工作节点ID
        if worker_id < 0 or worker_id >= 1024:
            raise ValueError("Worker ID must be between 0 and 1023")
        
        self.worker_id = worker_id
        self.epoch = epoch
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()
    
    def next_id(self) -> int:
        """
        生成下一个大整数ID
        """
        with self.lock:
            timestamp = self._current_timestamp()
            
            if timestamp < self.last_timestamp:
                # 时钟回拨处理
                raise ValueError(f"Clock moved backwards. Refusing to generate ID for {self.last_timestamp - timestamp} milliseconds")
            
            if timestamp == self.last_timestamp:
                # 同一毫秒内递增序列号
                self.sequence = (self.sequence + 1) & 0xFFF  # 12位序列号
                if self.sequence == 0:
                    # 序列号用尽，等待下一毫秒
                    timestamp = self._wait_next_millis(self.last_timestamp)
            else:
                # 新时间戳，重置序列号
                self.sequence = 0
            
            self.last_timestamp = timestamp
            
            # 组合各部分生成大整数ID
            return self._combine(timestamp, self.worker_id, self.sequence)
    
    def _current_timestamp(self) -> int:
        """获取当前时间戳 (毫秒)"""
        return int(time.time() * 1000) - self.epoch
    
    def _wait_next_millis(self, last_timestamp: int) -> int:
        """等待直到下一毫秒"""
        timestamp = self._current_timestamp()
        while timestamp <= last_timestamp:
            time.sleep(0.001)
            timestamp = self._current_timestamp()
        return timestamp
    
    def _combine(self, timestamp: int, worker_id: int, sequence: int) -> int:
        """
        组合时间戳、工作节点ID和序列号形成大整数ID
        
        使用动态位分配，支持更大的时间戳范围
        """
        # 时间戳部分 (动态位数)
        # 默认使用64位时间戳 (可支持约584,942,417年)
        timestamp_bits = 64
        
        # 工作节点ID (10位, 0-1023)
        worker_id_bits = 10
        
        # 序列号 (12位, 0-4095)
        sequence_bits = 12
        
        # 计算总位数
        total_bits = timestamp_bits + worker_id_bits + sequence_bits
        
        # 组合ID
        bigint_id = (timestamp << (worker_id_bits + sequence_bits)) | \
                    (worker_id << sequence_bits) | \
                    sequence
        
        return bigint_id
    
    @staticmethod
    def parse_id(bigint_id: int) -> Tuple[int, int, int]:
        """
        解析大整数ID为组成元素
        
        返回: (timestamp, worker_id, sequence)
        """
        # 序列号掩码 (低12位)
        sequence_mask = 0xFFF
        sequence = bigint_id & sequence_mask
        
        # 工作节点ID掩码 (中间10位)
        worker_id_mask = 0x3FF
        worker_id = (bigint_id >> 12) & worker_id_mask
        
        # 时间戳部分 (剩余高位)
        timestamp = bigint_id >> (12 + 10)
        
        return timestamp, worker_id, sequence

# 使用示例
if __name__ == "__main__":
    # 创建生成器
    generator = BigIntSnowflakeGenerator(worker_id=42)
    
    # 生成ID
    bigint_id = generator.next_id()
    print(f"Generated BigInt ID: {bigint_id}")
    
    # 解析ID
    timestamp, worker_id, sequence = BigIntSnowflakeGenerator.parse_id(bigint_id)
    print(f"Timestamp: {timestamp} ms since epoch")
    print(f"Worker ID: {worker_id}")
    print(f"Sequence: {sequence}")
    
    # 测试大整数支持
    print(f"ID is a big integer: {isinstance(bigint_id, int)}")
    print(f"ID size: {bigint_id.bit_length()} bits")