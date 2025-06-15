from pathlib import Path
import pandas as pd
import re
from fuzzywuzzy import fuzz
from typing import Dict, List, Optional

class CSVVerificationSystem:
    def __init__(self, csv_directory_or_files):
        self.csv_sources = csv_directory_or_files
        self.plans_df = None
        self.plan_database = {}
        self.load_plans_from_csv()
    
    def load_plans_from_csv(self):
        try:
            csv_files = []
            
            if isinstance(self.csv_sources, str):
                csv_path = Path(self.csv_sources)
                if csv_path.is_dir():
                    csv_files = list(csv_path.glob("*.csv"))
                elif csv_path.is_file():
                    csv_files = [csv_path]
                else:
                    return
            else:
                csv_files = [Path(f) for f in self.csv_sources]
            
            if not csv_files:
                return
            
            all_dataframes = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    if 'plans' in csv_file.name.lower():
                        df['source_file'] = 'plans'
                        df['data_type'] = 'plan'
                    elif 'coupon' in csv_file.name.lower():
                        df['source_file'] = 'coupons' 
                        df['data_type'] = 'coupon'
                    elif 'service' in csv_file.name.lower():
                        df['source_file'] = 'services'
                        df['data_type'] = 'service'
                    else:
                        df['source_file'] = csv_file.stem
                        df['data_type'] = 'unknown'
                    
                    all_dataframes.append(df)
                    
                except Exception:
                    continue
            
            if not all_dataframes:
                return
            
            self.plans_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
            self._build_search_database()
            
        except Exception:
            pass
    
    def _build_search_database(self):
        for _, row in self.plans_df.iterrows():
            name_candidates = []
            for col in ['plan_name', 'name', 'product_name', 'service_name', 'title']:
                if col in row and pd.notna(row[col]):
                    name_candidates.append(str(row[col]).strip())
            
            if not name_candidates:
                continue
                
            item_name = name_candidates[0]  
            
            if not item_name or item_name == 'nan':
                continue
            
            data_type = row.get('data_type', 'plan')
            
            if data_type == 'plan':
                item_info = {
                    'id': row.get('plan_id', row.get('id', '')),
                    'name': item_name,
                    'price': int(row['price']) if 'price' in row and pd.notna(row['price']) else 0,
                    'data': self._format_data(row.get('month_data')),
                    'calls': str(row['call_usage']) if 'call_usage' in row and pd.notna(row['call_usage']) else '',
                    'sms': str(row['message_usage']) if 'message_usage' in row and pd.notna(row['message_usage']) else '',
                    'benefit': str(row['benefit']) if 'benefit' in row and pd.notna(row['benefit']) else '',
                    'age_code': row.get('age_code', ''),
                    'category_code': row.get('category_code', ''),
                    'type': 'plan',
                    'source': row.get('source_file', 'unknown')
                }
            elif data_type in ['coupon', 'service']:
                item_info = {
                    'id': row.get('id', ''),
                    'name': item_name,
                    'price': int(row['price']) if 'price' in row and pd.notna(row['price']) else 0,
                    'data': '',
                    'calls': '',
                    'sms': '',
                    'benefit': str(row.get('description', row.get('benefit', ''))),
                    'age_code': '',
                    'category_code': '',
                    'type': data_type,
                    'source': row.get('source_file', 'unknown')
                }
            else:
                item_info = {
                    'id': row.get('id', ''),
                    'name': item_name,
                    'price': int(row['price']) if 'price' in row and pd.notna(row['price']) else 0,
                    'data': '',
                    'calls': '',
                    'sms': '',
                    'benefit': '',
                    'age_code': '',
                    'category_code': '',
                    'type': 'other',
                    'source': row.get('source_file', 'unknown')
                }
            
            self.plan_database[item_name.lower()] = item_info
            
            words = item_name.replace('(', ' ').replace(')', ' ').split()
            for word in words:
                if len(word) > 1:
                    self.plan_database[word.lower()] = item_info
    
    def _format_data(self, data_value):
        if pd.isna(data_value):
            return "정보없음"
        elif data_value == -1:
            return "무제한"
        elif data_value == 0:
            return "없음"
        else:
            return f"{data_value}GB"
    
    def verify_plan_exists(self, plan_name: str) -> Dict:
        # print(plan_name)
        plan_name_clean = plan_name.strip().lower()
        
        if plan_name_clean in self.plan_database:
            exact_match = self.plan_database[plan_name_clean]
            return {
                'exists': True,
                'confidence': 1.0,
                'match_type': 'exact',
                'matched_plan': exact_match,
                'original_input': plan_name
            }
        
        best_match = None
        best_score = 0
        
        unique_plans = {}
        for plan_info in self.plan_database.values():
            unique_plans[plan_info['name']] = plan_info
        
        for actual_name, plan_info in unique_plans.items():
            ratio_score = fuzz.ratio(plan_name_clean, actual_name.lower()) / 100
            partial_score = fuzz.partial_ratio(plan_name_clean, actual_name.lower()) / 100
            token_score = fuzz.token_sort_ratio(plan_name_clean, actual_name.lower()) / 100
            
            combined_score = (ratio_score * 0.4 + partial_score * 0.3 + token_score * 0.3)
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = plan_info
        
        if best_score >= 0.8:
            match_type = 'high_similarity'
        elif best_score >= 0.6:
            match_type = 'medium_similarity'
        elif best_score >= 0.4:
            match_type = 'low_similarity'
        else:
            match_type = 'no_match'
        
        return {
            'exists': best_score >= 0.6,
            'confidence': best_score,
            'match_type': match_type,
            'matched_plan': best_match,
            'original_input': plan_name
        }
    
    def verify_plan_details(self, plan_name: str, mentioned_price: int = None, 
                           mentioned_data: str = None) -> Dict:
        verification = self.verify_plan_exists(plan_name)
        
        if not verification['exists']:
            return verification
        
        matched_plan = verification['matched_plan']
        discrepancies = []
        
        if mentioned_price and matched_plan['price'] != mentioned_price:
            discrepancies.append(f"가격 불일치: 언급된 {mentioned_price:,}원 vs 실제 {matched_plan['price']:,}원")
        
        if mentioned_data and mentioned_data.lower() != matched_plan['data'].lower():
            discrepancies.append(f"데이터 불일치: 언급된 '{mentioned_data}' vs 실제 '{matched_plan['data']}'")
        
        verification['discrepancies'] = discrepancies
        verification['accuracy'] = 1.0 if not discrepancies else max(0.3, 1.0 - len(discrepancies) * 0.2)
        
        return verification
    
    def check_plan_hallucination(self, recommended_plan: str, recommendation_content: str, retriever=None) -> Dict:
        verification = self.verify_plan_exists(recommended_plan)
        
        mentioned_price = self._extract_price_from_text(recommendation_content)
        mentioned_data = self._extract_data_from_text(recommendation_content)
        
        detailed_verification = self.verify_plan_details(recommended_plan, mentioned_price, mentioned_data)
        
        return {
            "plan_exists": detailed_verification['exists'],
            "confidence_score": detailed_verification['confidence'],
            "matched_plan": detailed_verification['matched_plan']['name'] if detailed_verification['matched_plan'] else None,
            "discrepancies": detailed_verification.get('discrepancies', []),
            "evidence": [
                f"CSV 직접 검증 결과: {detailed_verification['match_type']}",
                f"신뢰도: {detailed_verification['confidence']:.2f}"
            ]
        }
    
    def _extract_price_from_text(self, text: str) -> Optional[int]:
        price_patterns = [
            r'(\d{1,3}(?:,\d{3})*)\s*원',
            r'월\s*(\d{1,3}(?:,\d{3})*)\s*원',
            r'(\d{1,3}(?:,\d{3})*)\s*원.*월'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text)
            if match:
                price_str = match.group(1).replace(',', '')
                try:
                    return int(price_str)
                except:
                    continue
        return None
    
    def _extract_data_from_text(self, text: str) -> Optional[str]:
        if re.search(r'무제한|unlimited', text, re.IGNORECASE):
            return '무제한'
        
        gb_match = re.search(r'(\d+)\s*GB', text, re.IGNORECASE)
        if gb_match:
            return f"{gb_match.group(1)}GB"
        
        return None
    
    def find_plans_by_criteria(self, price_range: tuple = None, data_min: int = None, 
                              age_code: str = None) -> List[Dict]:
        results = []
        
        unique_plans = {}
        for plan_info in self.plan_database.values():
            unique_plans[plan_info['name']] = plan_info
        
        for plan_info in unique_plans.values():
            if price_range:
                if not (price_range[0] <= plan_info['price'] <= price_range[1]):
                    continue
            
            if data_min and plan_info['data'] != '무제한':
                try:
                    data_gb = int(plan_info['data'].replace('GB', ''))
                    if data_gb < data_min:
                        continue
                except:
                    continue
            
            if age_code and plan_info['age_code'] != age_code:
                continue
            
            results.append(plan_info)
        
        return results
    
    def find_mentioned_plans(self, text: str) -> List[str]:
        mentioned_plans = []
        
        unique_plans = set(info['name'] for info in self.plan_database.values())
        
        for plan_name in unique_plans:
            plan_words = plan_name.lower().split()
            text_lower = text.lower()
            
            if len(plan_words) >= 2:
                if all(word in text_lower for word in plan_words[:2]):
                    mentioned_plans.append(plan_name)
            elif plan_name.lower() in text_lower:
                mentioned_plans.append(plan_name)
        
        return mentioned_plans
    
    def get_verification_status_message(self, confidence_score: float) -> str:
        if confidence_score >= 0.9:
            return "높은 신뢰도 - 정확한 정보"
        elif confidence_score >= 0.7:
            return "보통 신뢰도 - 대체로 정확"
        elif confidence_score >= 0.5:
            return "낮은 신뢰도 - 일부 불일치 가능"
        else:
            return "매우 낮은 신뢰도 - 정보 확인 필요"
    
    def get_plan_database_info(self) -> Dict:
        if not self.plan_database:
            return {
                "total_plans": 0,
                "total_entries": 0,
                "sample_plans": []
            }
        
        unique_items = {}
        for info in self.plan_database.values():
            item_name = info['name']
            if item_name not in unique_items:
                unique_items[item_name] = info
        
        return {
            "total_plans": len(unique_items),
            "total_entries": len(self.plan_database),
            "sample_plans": list(unique_items.keys())[:5]
        }
    
    def get_verification_summary(self) -> Dict:
        if not self.plan_database:
            return {
                'total_plans': 0,
                'search_keys': 0,
                'age_distribution': {},
                'price_range': {'min': 0, 'max': 0}
            }
        
        unique_items = {}
        for info in self.plan_database.values():
            item_name = info['name']
            if item_name not in unique_items:
                unique_items[item_name] = info
        
        age_distribution = {}
        prices = []
        
        for plan_info in unique_items.values():
            age_code = plan_info.get('age_code', '004')
            if age_code:
                age_distribution[age_code] = age_distribution.get(age_code, 0) + 1
            
            price = plan_info.get('price', 0)
            if price > 0:
                prices.append(price)
        
        return {
            'total_plans': len(unique_items),
            'search_keys': len(self.plan_database),
            'age_distribution': age_distribution,
            'price_range': {
                'min': min(prices) if prices else 0,
                'max': max(prices) if prices else 0
            }
        }